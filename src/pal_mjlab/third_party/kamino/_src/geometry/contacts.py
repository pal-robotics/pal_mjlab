# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Defines the representation of discrete contacts in Kamino.

This module provides a set of data types and operations that define
the data layout and conventions used to represent discrete contacts
within the Kamino solver. It includes:

- The :class:`ContactsKaminoData` dataclass defining the structure of contact data.

- The :class:`ContactMode` enumeration defining the discrete contact modes
and a member function that generates Warp functions to compute the contact
mode based on local contact velocities.

- Utility functions for constructing contact-local coordinate frames
supporting both a Z-up and X-up convention.

- The :class:`ContactsKamino` container which provides a high-level interface to
  manage contact data, including allocations, access, and common operations,
  and fundamentally serves as the primary output of collision detectors
  as well as a cache of contact data to warm-start physics solvers.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

import warp as wp

from ..core.bodies import is_immovable_for_kamino
from ..core.materials import MaterialMixMode, make_get_mixed_material_pair_property
from ..core.model import ModelKamino
from ..core.types import (
    to_warp_int32_array,
)
from ..utils import logger as msg
from .keying import build_pair_key2

###
# Module interface
###

__all__ = [
    "DEFAULT_GEOM_PAIR_CONTACT_GAP",
    "DEFAULT_GEOM_PAIR_MAX_CONTACTS",
    "DEFAULT_TRIANGLE_MAX_PAIRS",
    "DEFAULT_WORLD_MAX_CONTACTS",
    "ContactMode",
    "ContactsKamino",
    "ContactsKaminoData",
    "make_contact_frame_xnorm",
    "make_contact_frame_znorm",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


###
# Constants
###

DEFAULT_WORLD_MAX_CONTACTS: int = 128
"""
The global default for maximum number of contacts per world.
Used when allocating contact data without a specified capacity.
Set to `128`.
"""

DEFAULT_GEOM_PAIR_MAX_CONTACTS: int = 12
"""
The global default for maximum number of contacts per geom-pair.
Used when allocating contact data without a specified capacity.
Ignored for mesh-based collisions.
Set to `12` (with box-box collisions being a prototypical case).
"""

DEFAULT_TRIANGLE_MAX_PAIRS: int = 1_000_000
"""
The global default for maximum number of triangle pairs to consider in the narrow-phase.
Used only when the model contains triangle meshes or heightfields.
Defaults to `1_000_000`.
"""

DEFAULT_GEOM_PAIR_CONTACT_GAP: float = 1e-5
"""
The global default for the per-geometry detection gap [m].
Applied as a floor to each per-geometry gap value during pipeline
initialization so that every geometry has at least this detection
threshold.
Set to `1e-5`.
"""

DEFAULT_CULL_SPECULATIVE_CONTACTS: bool = True
"""
The global default for culling speculative contacts (with positive
margin-shifted distance) during Newton->Kamino conversion.
"""


###
# Types
###


class ContactMode(IntEnum):
    """An enumeration of discrete-contact modes."""

    ###
    # Contact Modes
    ###

    INACTIVE = -1
    """Indicates that contact is inactive (i.e. separated)."""

    OPENING = 0
    """Indicates that contact was previously closed (i.e. STICKING or SLIDING) and is now opening."""

    STICKING = 1
    """Indicates that contact is persisting (i.e. closed) without relative tangential motion."""

    SLIDING = 2
    """Indicates that contact is persisting (i.e. closed) with relative tangential motion."""

    ###
    # Utility Constants
    ###

    DEFAULT_VN_MIN = 1e-3
    """The minimum normal velocity threshold for determining contact open or closed modes."""

    DEFAULT_VT_MIN = 1e-3
    """The minimum tangential velocity threshold for determining contact stick or slip modes."""

    ###
    # Utility Functions
    ###

    @staticmethod
    def make_compute_mode_func(vn_tol: float = DEFAULT_VN_MIN, vt_tol: float = DEFAULT_VT_MIN):
        # Ensure tolerances are non-negative
        if vn_tol < 0.0:
            raise ValueError("ContactMode: vn_tol must be non-negative")
        if vt_tol < 0.0:
            raise ValueError("ContactMode: vt_tol must be non-negative")

        # Generate the compute mode function based on the specified tolerances
        @wp.func
        def _compute_mode(v: wp.vec3f) -> wp.int32:
            """
            Computes the discrete contact mode based on the contact velocity.

            Args:
                v: The contact velocity expressed in the local contact frame.

            Returns:
                The discrete contact mode as an integer value.
            """
            # Decompose the velocity into the normal and tangential components
            v_N = v.z
            v_T_norm = wp.sqrt(v.x * v.x + v.y * v.y)

            # Determine the contact mode
            mode = wp.int32(ContactMode.OPENING)
            if v_N <= wp.float32(vn_tol):
                if v_T_norm <= wp.float32(vt_tol):
                    mode = ContactMode.STICKING
                else:
                    mode = ContactMode.SLIDING

            # Return the resulting contact mode integer
            return mode

        # Return the generated compute mode function
        return _compute_mode


@dataclass
class ContactsKaminoData:
    """
    An SoA-based container to hold time-varying contact data of a set of contact elements.

    This container is intended as the final output of collision detectors and as input to solvers.
    """

    @staticmethod
    def _default_num_world_max_contacts() -> list[int]:
        return [0]

    model_max_contacts_host: int = 0
    """
    Host-side cache of the maximum number of contacts allocated across all worlds.
    Intended for managing data allocations and setting thread sizes in kernels.
    """

    world_max_contacts_host: list[int] = field(default_factory=_default_num_world_max_contacts)
    """
    Host-side cache of the maximum number of contacts allocated per world.
    Intended for managing data allocations and setting thread sizes in kernels.
    """

    model_max_contacts: wp.array[wp.int32] | None = None
    """
    The number of contacts pre-allocated across all worlds in the model.
    Shape of ``(1,)``.
    """

    model_active_contacts: wp.array[wp.int32] | None = None
    """
    The number of active contacts detected across all worlds in the model.
    Shape of ``(1,)``.
    """

    world_max_contacts: wp.array[wp.int32] | None = None
    """
    The maximum number of contacts pre-allocated for each world.
    Shape of ``(num_worlds,)``.
    """

    world_active_contacts: wp.array[wp.int32] | None = None
    """
    The number of active contacts detected in each world.
    Shape of ``(num_worlds,)``.
    """

    _contact_overflow_warning_emitted: wp.array[wp.int32] | None = None
    """Internal flag to track contact-buffer overflow warning messages."""

    wid: wp.array[wp.int32] | None = None
    """
    The world index of each active contact.
    Shape of ``(model_max_contacts_host,)``.
    """

    cid: wp.array[wp.int32] | None = None
    """
    The contact index of each active contact w.r.t its world.
    Shape of ``(model_max_contacts_host,)``.
    """

    gid_AB: wp.array[wp.vec2i] | None = None
    """
    The geometry indices of the geometry-pair AB associated with each active contact.
    Shape of ``(model_max_contacts_host,)``.
    """

    bid_AB: wp.array[wp.vec2i] | None = None
    """
    The body indices of the body-pair AB associated with each active contact.
    Shape of ``(model_max_contacts_host,)``.
    """

    position_A: wp.array[wp.vec3f] | None = None
    """
    The position of each active contact on the associated body-A in world coordinates.
    Shape of ``(model_max_contacts_host,)``.
    """

    position_B: wp.array[wp.vec3f] | None = None
    """
    The position of each active contact on the associated body-B in world coordinates.
    Shape of ``(model_max_contacts_host,)``.
    """

    gapfunc: wp.array[wp.vec4f] | None = None
    """
    Gap-function of each active contact, format ``(xyz: normal, w: signed_distance)``.
    The ``w`` component stores the signed distance between margin-shifted surfaces:
    negative means penetration past the resting separation, positive means separation
    within the detection gap.
    Shape of ``(model_max_contacts_host,)``.
    """

    frame: wp.array[wp.quatf] | None = None
    """
    The coordinate frame of each active contact as a rotation quaternion w.r.t the world.
    Shape of ``(model_max_contacts_host,)``.
    """

    material: wp.array[wp.vec2f] | None = None
    """
    The material properties of each active contact with format `(0: friction, 1: restitution)`.
    Shape of ``(model_max_contacts_host,)``.
    """

    margins: wp.array[wp.vec2f] | None = None
    """
    The shape-pair margins of each active contact.
    Shape of ``(model_max_contacts_host,)``.
    """

    key: wp.array[wp.uint64] | None = None
    """
    Integer key uniquely identifying each active contact.
    The per-contact key assignment is implementation-dependent, but is typically
    computed from the A/B geom-pair index as well as additional information such as:
    - the triangle index
    - shape-specific topological data
    - contact index w.r.t the geom-pair
    Shape of ``(model_max_contacts_host,)``.
    """

    reaction: wp.array[wp.vec3f] | None = None
    """
    The 3D contact reaction (force/impulse) expressed in the respective local contact frame.
    This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.
    Shape of ``(model_max_contacts_host,)``.
    """

    velocity: wp.array[wp.vec3f] | None = None
    """
    The 3D contact velocity expressed in the respective local contact frame.
    This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.
    Shape of ``(model_max_contacts_host,)``.
    """

    mode: wp.array[wp.int32] | None = None
    """
    The discrete contact mode expressed as an integer value.
    The possible values correspond to those of the :class:`ContactMode`.
    This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.
    Shape of ``(model_max_contacts_host,)``.
    """

    remap: wp.array[wp.int32] | None = None
    """
    Per-contact mapping back to the source contact index when converted from Newton :class:`Contacts`.
    Shape of ``(model_max_contacts_host,)``.

    Populated by :func:`convert_contacts_newton_to_kamino` so that each Kamino
    contact knows which original Newton contact it was generated from; entries
    default to ``-1`` for unmapped/inactive contacts.

    Consumed by :func:`convert_contacts_kamino_to_newton` along the
    existing-contacts path (``clear_output=False``) to write each contact's
    converted wrench back into the matching Newton slot. Only allocated when
    :class:`ContactsKamino` is constructed with ``remappable=True``.
    """

    def clear(self):
        """
        Clears the count of active contacts.
        """
        self.model_active_contacts.zero_()
        self.world_active_contacts.zero_()
        if self._contact_overflow_warning_emitted is not None:
            self._contact_overflow_warning_emitted.zero_()
        if self.remap is not None:
            self.remap.fill_(-1)

    def reset(self):
        """
        Clears the count of active contacts and resets contact data
        to sentinel values, indicating an empty set of contacts.
        """
        self.clear()
        self.wid.fill_(-1)
        self.cid.fill_(-1)
        self.gid_AB.fill_(wp.vec2i(-1, -1))
        self.bid_AB.fill_(wp.vec2i(-1, -1))
        self.mode.fill_(ContactMode.INACTIVE)
        self.reaction.zero_()
        self.velocity.zero_()


###
# Functions
###

COS_PI_6 = wp.constant(0.8660254037844387)


@wp.func
def make_contact_frame_znorm(n: wp.vec3f) -> wp.mat33f:
    n = wp.normalize(n)
    if wp.abs(n[0]) < COS_PI_6:
        e = wp.vec3f(1.0, 0.0, 0.0)
    else:
        e = wp.vec3f(0.0, 1.0, 0.0)
    o = wp.normalize(wp.cross(n, e))
    t = wp.normalize(wp.cross(o, n))
    return wp.mat33f(t.x, o.x, n.x, t.y, o.y, n.y, t.z, o.z, n.z)


@wp.func
def make_contact_frame_xnorm(n: wp.vec3f) -> wp.mat33f:
    n = wp.normalize(n)
    if wp.abs(n[0]) < COS_PI_6:
        e = wp.vec3f(1.0, 0.0, 0.0)
    else:
        e = wp.vec3f(0.0, 1.0, 0.0)
    o = wp.normalize(wp.cross(n, e))
    t = wp.normalize(wp.cross(o, n))
    return wp.mat33f(n.x, t.x, o.x, n.y, t.y, o.y, n.z, t.z, o.z)


@wp.func
def _print_world_contact_capacity_warning():
    wp.printf("Warning: Kamino per-world contact capacity exceeded. Increase the per-world contact capacity.\n")


@wp.func
def _print_model_contact_capacity_warning():
    wp.printf("Warning: Kamino model contact capacity exceeded. Increase the model contact capacity.\n")


@wp.func
def reserve_contact_capacity(
    model_max_contacts: wp.int32,
    world_max_contacts: wp.int32,
    wid: wp.int32,
    num_contacts: wp.int32,
    contact_model_num: wp.array[wp.int32],
    contact_world_num: wp.array[wp.int32],
    contact_overflow_warning_emitted: wp.array[wp.int32],
) -> wp.vec3i:
    """Reserve contact capacity atomically in the per-world and model counters.

    Args:
        model_max_contacts: Contact capacity of the model buffer.
        world_max_contacts: Contact capacity of world ``wid``.
        wid: World index of the contacts to reserve.
        num_contacts: Number of contacts requested.
        contact_model_num: Model-level active contact counter, shape ``(1,)``.
        contact_world_num: Per-world active contact counters, shape ``(num_worlds,)``.
        contact_overflow_warning_emitted: One-time overflow warning flag, shape ``(1,)``.

    Returns:
        ``(num_reserved, world_contact_index, model_index)``. ``num_reserved`` is ``0`` when no
        capacity was reserved, in which case both indices are ``-1``. It can be smaller than
        ``num_contacts``; callers requesting more than one contact must write only
        ``num_reserved`` contacts starting at ``world_contact_index`` and ``model_index``.
    """
    # Note: the world counter must be incremented first to ensure that once
    # a thread increments the global counter, it won't decrease it again after
    # because its world is saturated (leading to potential non-unique
    # mcid in other threads working on other worlds)
    # The decrease to the world counter if the model is saturated is not
    # problematic because the model is saturated for all threads in all worlds anyway.
    wcio = wp.atomic_add(contact_world_num, wid, num_contacts)
    if wcio >= world_max_contacts:
        wp.atomic_sub(contact_world_num, wid, num_contacts)
        if wp.atomic_exch(contact_overflow_warning_emitted, 0, 1) == 0:
            _print_world_contact_capacity_warning()
        return wp.vec3i(0, -1, -1)

    # Handle case where this thread saturated the world and only partial contacts can be written
    max_num_contacts = wp.min(world_max_contacts - wcio, num_contacts)
    if max_num_contacts < num_contacts:
        # Keep the world counter equal to the number of contacts retained.
        wp.atomic_sub(contact_world_num, wid, num_contacts - max_num_contacts)
        if wp.atomic_exch(contact_overflow_warning_emitted, 0, 1) == 0:
            _print_world_contact_capacity_warning()

    # Reserve model capacity only after the world reservation has been capped.
    mcio = wp.atomic_add(contact_model_num, 0, max_num_contacts)
    if mcio >= model_max_contacts:
        wp.atomic_sub(contact_model_num, 0, max_num_contacts)
        # Model saturation applies to every world, so this rollback is safe.
        wp.atomic_sub(contact_world_num, wid, max_num_contacts)
        if wp.atomic_exch(contact_overflow_warning_emitted, 0, 1) == 0:
            _print_model_contact_capacity_warning()
        return wp.vec3i(0, -1, -1)

    # Handle case where this thread saturated the model and only partial contacts can be written
    max_num_contacts_prev = max_num_contacts
    max_num_contacts = wp.min(model_max_contacts - mcio, max_num_contacts_prev)
    if max_num_contacts < max_num_contacts_prev:
        # Release capacity for contacts that cannot fit in the model buffer.
        wp.atomic_sub(contact_model_num, 0, max_num_contacts_prev - max_num_contacts)
        wp.atomic_sub(contact_world_num, wid, max_num_contacts_prev - max_num_contacts)
        if wp.atomic_exch(contact_overflow_warning_emitted, 0, 1) == 0:
            _print_model_contact_capacity_warning()

    return wp.vec3i(max_num_contacts, wcio, mcio)


###
# Interfaces
###


class ContactsKamino:
    """
    Provides a high-level interface to manage contact data,
    including allocations, access, and common operations.

    This container provides the primary output of collision detectors
    as well as a cache of contact data to warm-start physics solvers.
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        capacity: int | list[int] | None = None,
        default_max_contacts: int | None = None,
        device: wp.DeviceLike = None,
        remappable: bool = False,
    ):
        """
        Initializes a new ContactsKamino container.

        Args:
            model:
                The model container holding the time-invariant data of the system being simulated.
                If provided, the contacts will be finalized using the contact allocation meta-data of the model.
                Cannot be specified together with `capacity`.
                If `None``, and `capacity` is also `None`, the contacts will be created empty without
                allocating data, and can be finalized later by providing model/capacity to `finalize`.
            capacity:
                The maximum number of contacts to allocate if no model is provided.
                If an integer is provided, it specifies the capacity for a single world.
                If a list of integers is provided, it specifies the capacity for each world.
                Cannot be specified together with `model`.
            default_max_contacts:
                The default maximum number of contacts per world, if no model and no positive capacity
                are provided.
                If `None`, uses the default value of 128.
            device:
                The device on which to allocate the contacts data.
            remappable:
                Whether to allocate a buffer necessary for consistent mapping of Kamino and Newton
                contacts during conversions. Defaults to `False`, to be set to `True` if these
                contacts need to be converted from/to Newton contacts (e.g. if Kamino is used
                through the Newton API).
        """
        # Declare and initialize the default maximum number of contacts per world
        self._default_max_world_contacts: int = DEFAULT_WORLD_MAX_CONTACTS
        if default_max_contacts is not None:
            self._default_max_world_contacts = default_max_contacts

        # Cache the target device for all memory allocations
        self._device: wp.DeviceLike = None

        # Declare the contacts data container and initialize it to empty
        self._data: ContactsKaminoData = ContactsKaminoData()

        # If a capacity is specified, finalize the contacts data allocation
        if model is not None or capacity is not None:
            self.finalize(model=model, capacity=capacity, device=device, remappable=remappable)

    ###
    # Properties
    ###

    @property
    def default_max_world_contacts(self) -> int:
        """
        Returns the default maximum number of contacts per world.
        This value is used when the capacity at allocation-time is unspecified or equals 0.
        """
        return self._default_max_world_contacts

    @default_max_world_contacts.setter
    def default_max_world_contacts(self, max_contacts: int):
        """
        Sets the default maximum number of contacts per world.

        Args:
            max_contacts: The maximum number of contacts per world.
        """
        if max_contacts < 0:
            raise ValueError("max_contacts must be a non-negative integer")
        self._default_max_world_contacts = max_contacts

    @property
    def device(self) -> wp.DeviceLike:
        """
        Returns the device on which the contacts data is allocated.
        """
        return self._device

    @property
    def data(self) -> ContactsKaminoData:
        """
        Returns the managed contacts data container.
        """
        self._assert_has_data()
        return self._data

    @property
    def model_max_contacts_host(self) -> int:
        """
        Returns the host-side cache of the maximum number of contacts allocated across all worlds.
        Intended for managing data allocations and setting thread sizes in kernels.
        """
        self._assert_has_data()
        return self._data.model_max_contacts_host

    @property
    def world_max_contacts_host(self) -> list[int]:
        """
        Returns the host-side cache of the maximum number of contacts allocated per world.
        Intended for managing data allocations and setting thread sizes in kernels.
        """
        self._assert_has_data()
        return self._data.world_max_contacts_host

    @property
    def model_max_contacts(self) -> wp.array[wp.int32]:
        """
        Returns the maximum number contacts pre-allocated across all worlds in the model.
        Shape of ``(1,)``.
        """
        self._assert_has_data()
        return self._data.model_max_contacts

    @property
    def model_active_contacts(self) -> wp.array[wp.int32]:
        """
        Returns the number of active contacts detected across all worlds in the model.
        Shape of ``(1,)``.
        """
        self._assert_has_data()
        return self._data.model_active_contacts

    @property
    def world_max_contacts(self) -> wp.array[wp.int32]:
        """
        Returns the maximum number of contacts pre-allocated for each world.
        Shape of ``(num_worlds,)``.
        """
        self._assert_has_data()
        return self._data.world_max_contacts

    @property
    def world_active_contacts(self) -> wp.array[wp.int32]:
        """
        Returns the number of active contacts detected in each world.
        Shape of ``(num_worlds,)``.
        """
        self._assert_has_data()
        return self._data.world_active_contacts

    @property
    def wid(self) -> wp.array[wp.int32]:
        """
        Returns the world index of each active contact.
        Shape of ``(model_max_contacts_host,)``.
        """
        self._assert_has_data()
        return self._data.wid

    @property
    def cid(self) -> wp.array[wp.int32]:
        """
        Returns the contact index of each active contact w.r.t its world.
        Shape of ``(model_max_contacts_host,)``.
        """
        self._assert_has_data()
        return self._data.cid

    @property
    def gid_AB(self) -> wp.array[wp.vec2i]:
        """
        Returns the geometry indices of the geometry-pair AB associated with each active contact.
        Shape of ``(model_max_contacts_host,)``.
        """
        self._assert_has_data()
        return self._data.gid_AB

    @property
    def bid_AB(self) -> wp.array[wp.vec2i]:
        """
        Returns the body indices of the body-pair AB associated with each active contact.
        Shape of ``(model_max_contacts_host,)``.
        """
        self._assert_has_data()
        return self._data.bid_AB

    @property
    def position_A(self) -> wp.array[wp.vec3f]:
        """
        Returns the position of each active contact on the associated body-A in world coordinates.
        Shape of ``(model_max_contacts_host,)``.
        """
        self._assert_has_data()
        return self._data.position_A

    @property
    def position_B(self) -> wp.array[wp.vec3f]:
        """
        Returns the position of each active contact on the associated body-B in world coordinates.
        Shape of ``(model_max_contacts_host,)``.
        """
        self._assert_has_data()
        return self._data.position_B

    @property
    def gapfunc(self) -> wp.array[wp.vec4f]:
        """
        Returns the gap-function of each active contact, packed as``(xyz: normal, w: distance)``.
        Shape of ``(model_max_contacts_host,)``.

        The ``w`` component stores the signed ``distance`` between margin-shifted surfaces:
        - ``w < 0`` means penetration past the resting separation defined by the margin
        - ``w > 0`` means margin-shifted separation within the detection gap
        """
        self._assert_has_data()
        return self._data.gapfunc

    @property
    def frame(self) -> wp.array[wp.quatf]:
        """
        Returns the coordinate frame of each active contact as a rotation quaternion w.r.t the world.
        Shape of ``(model_max_contacts_host,)``.
        """
        self._assert_has_data()
        return self._data.frame

    @property
    def material(self) -> wp.array[wp.vec2f]:
        """
        Returns the material properties of each active contact with format `(0: friction, 1: restitution)`.
        Shape of ``(model_max_contacts_host,)``.
        """
        self._assert_has_data()
        return self._data.material

    @property
    def margins(self) -> wp.array[wp.vec2f]:
        """
        Returns the effective shape-pair margins of each active contact.
        Shape of ``(model_max_contacts_host,)``.
        """
        self._assert_has_data()
        return self._data.margins

    @property
    def key(self) -> wp.array[wp.uint64]:
        """
        Returns the integer key uniquely identifying each active contact.
        The per-contact key assignment is implementation-dependent, but is typically
        computed from the A/B geom-pair index as well as additional information such as:
        - the triangle index
        - shape-specific topological data
        - contact index w.r.t the geom-pair
        Shape of ``(model_max_contacts_host,)``.
        """
        self._assert_has_data()
        return self._data.key

    @property
    def reaction(self) -> wp.array[wp.vec3f]:
        """
        Returns the 3D contact reaction (force/impulse) expressed in the respective local contact frame.
        This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.
        Shape of ``(model_max_contacts_host,)``.
        """
        self._assert_has_data()
        return self._data.reaction

    @property
    def velocity(self) -> wp.array[wp.vec3f]:
        """
        Returns the 3D contact velocity expressed in the respective local contact frame.
        This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.
        Shape of ``(model_max_contacts_host,)``.
        """
        self._assert_has_data()
        return self._data.velocity

    @property
    def mode(self) -> wp.array[wp.int32]:
        """
        Returns the discrete contact mode expressed as an integer value.
        The possible values correspond to those of the :class:`ContactMode`.
        This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.
        Shape of ``(model_max_contacts_host,)``.
        """
        self._assert_has_data()
        return self._data.mode

    @property
    def remap(self) -> wp.array[wp.int32] | None:
        """
        Returns the remapped contact index of each active contact.
        Shape of ``(model_max_contacts_host,)``.
        """
        self._assert_has_data()
        return self._data.remap

    ###
    # Operations
    ###

    def finalize(
        self,
        model: ModelKamino | None = None,
        capacity: int | list[int] | None = None,
        device: wp.DeviceLike = None,
        remappable: bool = False,
    ):
        """
        Finalizes the contacts data allocations based on the specified model or capacity.

        Args:
            model:
                The model container holding the time-invariant data of the system being simulated.
                If provided, the contacts will be finalized using the contact allocation meta-data of the model.
                Cannot be specified together with `capacity`.
            capacity:
                The maximum number of contacts to allocate if no model is provided.
                If an integer is provided, it specifies the capacity for a single world.
                If a list of integers is provided, it specifies the capacity for each world.
                Cannot be specified together with `model`.
            device:
                The device on which to allocate the contacts data, if no model is provided.
            remappable:
                Whether to allocate a buffer necessary for consistent mapping of Kamino and Newton
                contacts during conversions. Defaults to `False`, to be set to `True` if these
                contacts need to be converted from/to Newton contacts (e.g. if Kamino is used
                through the Newton API).
        """
        # Raise errors if both model and capacity are provided or both are None
        if model is not None and capacity is not None:
            raise ValueError("Expected either 'model' or 'capacity' argument to be provided, but not both.")
        if model is None and capacity is None:
            raise ValueError("Expected either 'model' or 'capacity' argument to be provided, but got neither")

        # The memory allocation requires the total number of contacts (over multiple worlds)
        # as well as the contacts capacities for each world. Corresponding sizes are defaulted to 0 (empty).
        model_max_contacts = 0
        world_max_contacts = [0]

        # If a model is provided, extract the required contacts capacity from that
        if model is not None:
            model_max_contacts: int = 0
            world_max_contacts: list[int] = [0 for _ in range(model.size.num_worlds)]
            if model.geoms.model_minimum_contacts > 0:
                model_max_contacts = model.geoms.model_minimum_contacts
                world_max_contacts = model.geoms.world_minimum_contacts
            else:
                num_worlds = model.size.num_worlds
                world_max_contacts = [model_max_contacts // num_worlds] * num_worlds
            capacity = world_max_contacts
            self._device = model.device
        else:
            self._device = device

        # If the capacity is a list, this means we are allocating for multiple worlds
        if isinstance(capacity, list):
            if len(capacity) == 0:
                raise ValueError("`capacity` must be an non-empty list")
            for i in range(len(capacity)):
                if capacity[i] < 0:
                    raise ValueError(f"`capacity[{i}]` must be a non-negative integer")
                if capacity[i] == 0:
                    capacity[i] = self._default_max_world_contacts
            model_max_contacts = sum(capacity)
            world_max_contacts = capacity

        # If the capacity is a single integer, this means we are allocating for a single world
        elif isinstance(capacity, int):
            if capacity < 0:
                raise ValueError("`capacity` must be a non-negative integer")
            if capacity == 0:
                capacity = self._default_max_world_contacts
            model_max_contacts = capacity
            world_max_contacts = [capacity]

        else:
            raise TypeError("`capacity` must be an integer or a list of integers")

        # Skip allocation if there are no contacts to allocate
        if model_max_contacts == 0:
            msg.debug("ContactsKamino: Skipping contact data allocations since total requested capacity was `0`.")
            return

        # Allocate the contacts data on the specified device
        with wp.ScopedDevice(self._device):
            self._data = ContactsKaminoData(
                model_max_contacts_host=model_max_contacts,
                world_max_contacts_host=world_max_contacts,
                model_max_contacts=to_warp_int32_array([model_max_contacts]),
                model_active_contacts=wp.zeros(shape=1, dtype=wp.int32),
                world_max_contacts=to_warp_int32_array(world_max_contacts),
                world_active_contacts=wp.zeros(shape=len(world_max_contacts), dtype=wp.int32),
                _contact_overflow_warning_emitted=wp.zeros(shape=1, dtype=wp.int32),
                wid=wp.full(value=-1, shape=(model_max_contacts,), dtype=wp.int32),
                cid=wp.full(value=-1, shape=(model_max_contacts,), dtype=wp.int32),
                gid_AB=wp.full(value=wp.vec2i(-1, -1), shape=(model_max_contacts,), dtype=wp.vec2i),
                bid_AB=wp.full(value=wp.vec2i(-1, -1), shape=(model_max_contacts,), dtype=wp.vec2i),
                position_A=wp.zeros(shape=(model_max_contacts,), dtype=wp.vec3f),
                position_B=wp.zeros(shape=(model_max_contacts,), dtype=wp.vec3f),
                gapfunc=wp.zeros(shape=(model_max_contacts,), dtype=wp.vec4f),
                frame=wp.zeros(shape=(model_max_contacts,), dtype=wp.quatf),
                material=wp.zeros(shape=(model_max_contacts,), dtype=wp.vec2f),
                margins=wp.zeros(shape=(model_max_contacts,), dtype=wp.vec2f),
                key=wp.zeros(shape=(model_max_contacts,), dtype=wp.uint64),
                reaction=wp.zeros(shape=(model_max_contacts,), dtype=wp.vec3f),
                velocity=wp.zeros(shape=(model_max_contacts,), dtype=wp.vec3f),
                mode=wp.full(value=ContactMode.INACTIVE, shape=(model_max_contacts,), dtype=wp.int32),
                remap=wp.full(value=-1, shape=(model_max_contacts,), dtype=wp.int32) if remappable else None,
            )

    def clear(self):
        """
        Clears the count of active contacts.
        """
        self._assert_has_data()
        if self._data.model_max_contacts_host > 0:
            self._data.clear()

    def reset(self):
        """
        Clears the count of active contacts and resets data to sentinel values.
        """
        self._assert_has_data()
        if self._data.model_max_contacts_host > 0:
            self._data.reset()

    ###
    # Internals
    ###

    def _assert_has_data(self):
        if self._data.model_max_contacts_host == 0:
            raise RuntimeError("ContactsKaminoData has not been allocated. Call `finalize()` before accessing data.")


