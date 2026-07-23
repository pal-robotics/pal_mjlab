"""Script to launch a menu to start training or deploy policies"""
"""

Idea :

To have an interactive menu that makes it much easier to launch simulation and deploy policies without running and/or modifying bash commands

"""
import os
import sys
import subprocess
from dataclasses import is_dataclass, fields
from pathlib import Path

import tyro

import threading
import tkinter as tk
from tkinter import scrolledtext, simpledialog
import tkinter.font as tkfont
import re
import paramiko
import shlex

import mjlab
from mjlab.tasks.registry import load_env_cfg


class RemoteProcess:
  def __init__(self, user, hostname, password, pid):
    self.user = user
    self.hostname = hostname
    self.password = password
    self.pid = pid

  def kill(self):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
      ssh.connect(
        hostname=self.hostname,
        username=self.user,
        password=self.password,
        allow_agent=False,
        look_for_keys=False,
      )

      command = f"""
      get_children() {{
        for child in $(cat /proc/$1/task/$1/children 2>/dev/null); do
          get_children "$child"
          kill -TERM "$child" 2>/dev/null
        done
      }}

      get_children {self.pid}
      kill -TERM {self.pid} 2>/dev/null
      """

      stdin, stdout, stderr = ssh.exec_command(command)

      # Wait for remote kill command to finish
      stdout.channel.recv_exit_status()

    finally:
      ssh.close()


LOG_ROOT = Path("./logs").resolve()

def find_latest_checkpoints():
  checkpoints_dict = {}  # key = (exp_folder, run_folder), value = (step, path)

  if not LOG_ROOT.exists():
    return []

  for model_path in LOG_ROOT.rglob("model_*.pt"):
    match = re.match(r"model_(\d+)\.pt", model_path.name)
    if not match:
      continue

    step = int(match.group(1))
    run_folder = model_path.parent.name
    exp_folder = model_path.parent.parent.name

    key = (exp_folder, run_folder)
    if key not in checkpoints_dict or step > checkpoints_dict[key][0]:
      checkpoints_dict[key] = (step, model_path)

  # convert dict to list of tuples with label
  checkpoints = []
  for (exp_folder, run_folder), (step, path) in checkpoints_dict.items():
    label = f"{exp_folder}/{run_folder} (step {step})"
    checkpoints.append((label, path))

  # optional: sort by experiment/run name
  checkpoints.sort(key=lambda x: x[0])

  return checkpoints

SNAPSHOTS_DIR = Path("./training_snapshots").resolve()

def save_training_snapshot(environment_name, job_name, extra_opts, description = ""):
  SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
  from datetime import datetime
  date_str = datetime.now().strftime("%Y-%m-%d")
  timestamp_str = datetime.now().strftime("%H:%M:%S")
  snapshot_file = SNAPSHOTS_DIR / f"{date_str}.txt"
  with open(snapshot_file, "a") as f:
    f.write(f"  [{timestamp_str}]\n")
    f.write(f"env        : {environment_name}\n")
    f.write(f"job name   : {job_name}\n")
    f.write(f"options    : --env.scene.num-envs 4096 --agent.run-name {job_name} --agent.logger tensorboard --agent.save-interval 500 {extra_opts if extra_opts else ''}\n")
    f.write(f"________________________________________________\n")
    f.write(f"Description   :\n\n{description}\n")
    f.write(f"________________________________________________\n")
    f.write("\n\n")


REMOTE_CONFIG_PATH = Path("./remote_config.txt").resolve()

def load_remote_setup():
  try:
    if not REMOTE_CONFIG_PATH.exists():
      return "", "", "", 0

    with REMOTE_CONFIG_PATH.open("r") as f:
      lines = [line.strip() for line in f]

    # Must contain exactly 3 non-empty lines
    if len(lines) != 3:
      return "", "", "", 0

    user, hostname, remote_folder = lines

    if not user or not hostname or not remote_folder:
      return "", "", "", 0

    return user, hostname, remote_folder, 1

  except (OSError, UnicodeDecodeError):
    return "", "", "", 0


def write_remote_setup(user, hostname, remote_folder):
  with REMOTE_CONFIG_PATH.open("w") as f:
    f.write(f"{user}\n")
    f.write(f"{hostname}\n")
    f.write(f"{remote_folder}\n")


remote_setup = load_remote_setup()

tasks = []

# --- Process registry for cleanup ---
processes = {}

def kill_proc_tree(proc):
  # Remote process
  if isinstance(proc, RemoteProcess):
    try:
      proc.kill()
    except Exception as e:
      print(f"Error killing remote process: {e}")
    return
  try:
    # Get all child PIDs via /proc filesystem (Linux only)
    def get_children(pid):
      children = []
      try:
        task_dir = Path(f"/proc/{pid}/task")
        for tid in task_dir.iterdir():
          children_file = tid / "children"
          if children_file.exists():
            pids = children_file.read_text().split()
            for child_pid in pids:
              child_pid = int(child_pid)
              children.append(child_pid)
              children.extend(get_children(child_pid))
      except (FileNotFoundError, PermissionError):
        pass
      return children

    children = get_children(proc.pid)

    # Terminate children first, then parent
    for pid in children:
      try:
        os.kill(pid, signal.SIGTERM)
      except ProcessLookupError:
        pass

    proc.terminate()
    proc.wait(timeout=5)

  except subprocess.TimeoutExpired:
    # Force kill anything still alive
    for pid in children:
      try:
        os.kill(pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
    proc.kill()

  except Exception as e:
    print(f"Error killing process tree: {e}")

def cleanup(*args):
  for name, proc in list(processes.items()):
    print(f"Terminating {name}...")
    try:
      kill_proc_tree(proc)
    except Exception as e:
      print(f"Error terminating {name}: {e}")
    finally:
      processes.pop(name, None)
  sys.exit(0)

import atexit, signal
atexit.register(cleanup)
signal.signal(signal.SIGTERM, cleanup)


MAX_CONSOLE_LINES = 500

def append_to_console(console, text, tag=None):
  if tag:
    console.insert(tk.END, text, tag)
  else:
    console.insert(tk.END, text)
  
  # Count lines and trim oldest if over limit
  line_count = int(console.index(tk.END).split(".")[0])
  if line_count > MAX_CONSOLE_LINES:
    excess = line_count - MAX_CONSOLE_LINES
    console.delete("1.0", f"{excess + 1}.0")
  
  console.see(tk.END)


def extract_default_reward_lines(env_cfg):
  """Best-effort extraction of default reward terms from a loaded env cfg.

  Returns a list of strings formatted as CLI-option "key value" pairs
  (without the leading "--", matching the format the train window's
  options box expects, e.g. "env.rewards.track_lin_vel_xy_exp.weight 1.0").

  This is intentionally defensive: reward containers across tasks may be a
  dict[str, RewardTermCfg], a dataclass of RewardTermCfg fields, or missing
  entirely, so we introspect rather than assume one exact shape.
  """
  lines = []

  rewards = getattr(env_cfg, "rewards", None)
  if rewards is None:
    return lines

  # Normalize to an iterable of (name, term) pairs.
  if isinstance(rewards, dict):
    items = list(rewards.items())
  elif is_dataclass(rewards):
    items = [(f.name, getattr(rewards, f.name)) for f in fields(rewards)]
  elif hasattr(rewards, "items"):
    items = list(rewards.items())
  else:
    items = []

  for name, term in items:
    if term is None:
      continue

    weight = getattr(term, "weight", None)
    if weight is not None:
      lines.append(f"env.rewards.{name}.weight {weight}")
      continue

    # Fallback: term has no obvious "weight" attribute, dump what we can.
    if is_dataclass(term):
      for f in fields(term):
        val = getattr(term, f.name)
        if isinstance(val, (int, float, str, bool)):
          lines.append(f"env.rewards.{name}.{f.name} {val}")
    elif isinstance(term, (int, float, str, bool)):
      lines.append(f"env.rewards.{name} {term}")

  return lines


# --- GUI ---
def open_menu():
    BG       = "#0f1117"
    PANEL    = "#1a1d27"
    ACCENT   = "#00d4aa"
    ACCENT2  = "#7c6af7"
    TEXT     = "#e8eaf0"
    MUTED    = "#8a8fa6"
    DANGER   = "#ff5a5a"

    # --- Core launcher ---
    def launch_process(
      command,
      menu_console,
      label="process",
      on_complete=None,
      extra_env=None,
    ):
      def run():
        env = os.environ.copy()

        if extra_env:
          env.update(extra_env)

        proc = subprocess.Popen(
          command,
          shell=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
          executable="/bin/bash",
          env=env,
        )

        processes[label] = proc

        menu_console.after(
          0,
          append_to_console,
          menu_console,
          f"$ {command}\n",
          "cmd",
        )

        menu_console.after(0, refresh_process_list)

        output = []

        for line in proc.stdout:
          output.append(line)

          menu_console.after(
            0,
            append_to_console,
            menu_console,
            line,
          )

        proc.wait()

        status = (
          "\n✓ done"
          if proc.returncode == 0
          else f"\n✗ exited {proc.returncode}"
        )

        menu_console.after(
          0,
          append_to_console,
          menu_console,
          f"{status}\n\n",
          "status",
        )

        processes.pop(label, None)

        menu_console.after(
          0,
          refresh_process_list,
        )

        if on_complete:
          menu_console.after(
            0,
            on_complete,
            "".join(output),
          )


      def run_remote():
        user, ip, remote_path, remote_flag = remote_setup

        if remote_flag != 1:
          menu_console.after(
            0,
            append_to_console,
            menu_console,
            "[ERROR] Remote setup has not been done !\n",
            "status",
          )
          return

        # Ask password on Tkinter main thread
        password = [None]
        event = threading.Event()

        def ask_password_and_signal():
          password[0] = simpledialog.askstring(
            "SSH Password",
            f"Enter SSH password for {user}:",
            show="*",
            parent=root,
          )
          event.set()

        root.after(0, ask_password_and_signal)
        event.wait()

        if password[0] is None:
            return

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(
            paramiko.AutoAddPolicy()
        )

        try:
          ssh.connect(
            hostname=ip,
            username=user,
            password=password[0],
            allow_agent=False,
            look_for_keys=False,
          )

          menu_console.after(
            0,
            append_to_console,
            menu_console,
            f"$ ssh {user}@{ip}\n",
            "cmd",
          )

          # The remote shell PID becomes the command PID through exec.
          remote_command = (
            f"cd {shlex.quote(remote_path)} && "
            "echo __REMOTE_PID=$$ && "
            f"exec bash -lc {shlex.quote(command)}"
          )

          stdin, stdout, stderr = ssh.exec_command(
            remote_command
          )

          # Read PID marker
          pid_line = stdout.readline().strip()

          if not pid_line.startswith("__REMOTE_PID="):
            raise RuntimeError(
              f"Failed to obtain remote PID: {pid_line}"
            )

          pid = int(
            pid_line.split("=", 1)[1]
          )

          remote_proc = RemoteProcess(
            user=user,
            hostname=ip,
            password=password[0],
            pid=pid,
          )

          # Register immediately
          processes[label] = remote_proc

          menu_console.after(
            0,
            refresh_process_list,
          )

          output = []

          # Read stdout
          for line in iter(stdout.readline, ""):
            output.append(line)

            menu_console.after(
              0,
              append_to_console,
              menu_console,
              line,
            )

          # Read stderr
          for line in iter(stderr.readline, ""):
            output.append(line)

            menu_console.after(
              0,
              append_to_console,
              menu_console,
              line,
              "status",
            )

          # Wait for actual command exit
          exit_status = stdout.channel.recv_exit_status()

          status = (
            "\n✓ done"
            if exit_status == 0
            else f"\n✗ exited {exit_status}"
          )

          menu_console.after(
            0,
            append_to_console,
            menu_console,
            f"{status}\n\n",
            "status",
          )

          if on_complete:
            menu_console.after(
              0,
              on_complete,
              "".join(output),
            )

        except paramiko.AuthenticationException:
          menu_console.after(
            0,
            append_to_console,
            menu_console,
            "[ERROR] SSH authentication failed.\n",
            "status",
          )

        except paramiko.SSHException as e:
          menu_console.after(
            0,
            append_to_console,
            menu_console,
            f"[ERROR] SSH connection failed: {e}\n",
            "status",
          )

        except Exception as e:
          menu_console.after(
            0,
            append_to_console,
            menu_console,
            f"[ERROR] Remote process failed: {e}\n",
            "status",
          )

        finally:
          ssh.close()

          # Remove process from registry when it finishes
          processes.pop(label, None)

          menu_console.after(
            0,
            refresh_process_list,
          )


      if remote_var.get():
        thread = threading.Thread(
          target=run_remote,
          daemon=True,
        )
      else:
        thread = threading.Thread(
          target=run,
          daemon=True,
        )

      thread.start()

      return thread

    root = tk.Tk()
    root.title("mjlab launcher")
    root.configure(bg=BG)
    root.geometry("1600x900")
    root.resizable(False, False)
    root.minsize(700, 450)

    # Fonts
    title_font   = tkfont.Font(family="Courier New", size=13, weight="bold")
    label_font   = tkfont.Font(family="Courier New", size=10)
    console_font = tkfont.Font(family="Courier New", size=10)
    btn_font     = tkfont.Font(family="Courier New", size=10, weight="bold")

    # ── Layout: left panel + right console ──────────────────────────────────
    root.columnconfigure(0, weight=0)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)

    # Left panel
    left = tk.Frame(root, bg=PANEL, width=260)
    left.grid(row=0, column=0, sticky="ns", padx=(10, 4), pady=10)
    left.grid_propagate(False)

    tk.Label(left, text="PAL_MJLAB", font=title_font,
             bg=PANEL, fg=ACCENT).pack(anchor="w", padx=16, pady=(18, 4))
    tk.Label(left, text="simulation launcher", font=label_font,
             bg=PANEL, fg=MUTED).pack(anchor="w", padx=16, pady=(0, 20))

    tk.Frame(left, bg=MUTED, height=1).pack(fill="x", padx=16, pady=(0, 16))

    selected_console = tk.IntVar(value=0)  # default terminal 0
    terminal_names = ["Terminal 1", "Terminal 2", "Terminal 3", "Terminal 4"]

    def make_button(parent, text, color, command):
        btn = tk.Button(
            parent, text=text, font=btn_font,
            bg=color, fg=BG, activebackground=TEXT, activeforeground=BG,
            relief="flat", cursor="hand2", bd=0,
            padx=12, pady=8, command=command
        )
        btn.pack(fill="x", padx=16, pady=4)

        # Hover effect
        btn.bind("<Enter>", lambda e: btn.config(bg=TEXT))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
        return btn


    def run_list_envs():
      def on_complete(output):
        global tasks
        tasks = re.findall(r'\|\s*\d+\s*\|\s*([\w-]+)\s*\|', output)

      launch_process("uv run list-envs", consoles[selected_console.get()], label="list-envs", on_complete=on_complete)

    def run_tsp():
      launch_process("tsp", consoles[selected_console.get()], label="tsp")

    def run_tsp_t():
      launch_process("tsp -t", consoles[selected_console.get()], label="tsp -t")

    def run_tsp_remove():
      win = tk.Toplevel(root)
      win.title("Remove tsp Job")
      win.configure(bg=BG)
      win.geometry("360x180")
      win.resizable(False, False)

      tk.Label(win, text="tsp Job ID:", font=label_font,
               bg=BG, fg=MUTED).pack(anchor="w", padx=20, pady=(20, 6))
      id_entry = tk.Entry(win, font=label_font)
      id_entry.pack(fill="x", padx=20, pady=(0, 16))
      id_entry.focus_set()

      def confirm():
        job_id = id_entry.get().strip()
        if not job_id:
          append_to_console(consoles[selected_console.get()], "✗ No job id entered\n", "status")
          return
        win.destroy()
        launch_process(f"tsp -r {job_id}", consoles[selected_console.get()], label=f"tsp-r:{job_id}")

      id_entry.bind("<Return>", lambda e: confirm())

      btn = tk.Button(win, text="✕  Remove Job", font=btn_font,
                      bg=DANGER, fg=BG,
                      relief="flat", cursor="hand2",
                      padx=12, pady=8,
                      command=confirm)
      btn.pack(padx=20, pady=(0, 20))

    def run_deploy():
      if not tasks:
        append_to_console(consoles[selected_console.get()], "✗ No tasks loaded, click 'List Tasks' first\n", "status")
        return

      checkpoints = find_latest_checkpoints()
      if not checkpoints:
        append_to_console(consoles[selected_console.get()], "✗ No checkpoints found\n", "status")
        return

      # New window
      win = tk.Toplevel(root)
      win.title("Deploy Policy")
      win.configure(bg=BG)
      win.geometry("500x300")
      win.resizable(False, False)

      tk.Label(win, text="Select task to deploy", font=label_font,
                bg=BG, fg=MUTED).pack(anchor="w", padx=20, pady=(20, 6))

      selected_task = tk.StringVar(value=tasks[0] if tasks else "")
      dropdown_task = tk.OptionMenu(win, selected_task, *tasks)
      dropdown_task.pack(fill="x", padx=20, pady=(0, 16))

      # NEW: checkpoint selector
      tk.Label(win, text="Select latest checkpoint", font=label_font,
                bg=BG, fg=MUTED).pack(anchor="w", padx=20, pady=(0, 6))

      checkpoint_labels = [c[0] for c in checkpoints]
      selected_ckpt = tk.StringVar(value=checkpoint_labels[0])
      dropdown_ckpt = tk.OptionMenu(win, selected_ckpt, *checkpoint_labels)
      dropdown_ckpt.pack(fill="x", padx=20, pady=(0, 16))

      agentZero_check_var = tk.BooleanVar(value=False)
      agentRandom_check_var = tk.BooleanVar(value=False)

      checkbox = tk.Checkbutton(
          win,
          text="Agent Zero",
          variable=agentZero_check_var,
          onvalue=True,
          offvalue=False,
          font=label_font,
          bg=BG,
          fg="#e8eaf0",
          activebackground=BG,
          activeforeground="#00d4aa",  
          selectcolor=BG,               
          relief="flat",
          anchor="w",                   
          padx=5,
          pady=2,
          bd=0,
          highlightthickness=0,
      )
      checkbox.pack(fill="x", padx=20, pady=(0,16))

      checkbox = tk.Checkbutton(
          win,
          text="Agent Random",
          variable=agentRandom_check_var,
          onvalue=True,
          offvalue=False,
          font=label_font,
          bg=BG,
          fg="#e8eaf0",
          activebackground=BG,
          activeforeground="#00d4aa",  
          selectcolor=BG,               
          relief="flat",
          anchor="w",                   
          padx=5,
          pady=2,
          bd=0,
          highlightthickness=0,
      )
      checkbox.pack(fill="x", padx=20, pady=(0,16))

      def confirm():
        task = selected_task.get()
        label = selected_ckpt.get()

        if (agentZero_check_var.get()) :
          cmd = f"uv run play {task} --agent zero"
        elif (agentRandom_check_var.get()) :
          cmd = f"uv run play {task} --agent random"
        else :
          # find matching path
          ckpt_path = None
          for l, path in checkpoints:
            if l == label:
              ckpt_path = path
              break
          cmd = f"uv run play {task} --checkpoint-file {ckpt_path}"

        win.destroy()
        launch_process(
          cmd,
          consoles[selected_console.get()],
          label="deploy"
        )

      btn = tk.Button(win, text="▶  Deploy", font=btn_font,
                      bg=ACCENT, fg=BG,
                      relief="flat", cursor="hand2",
                      padx=12, pady=8,
                      command=confirm)
      btn.pack(padx=20, pady=(0, 20))

    def run_train():
      if not tasks:
        append_to_console(consoles[selected_console.get()], "✗ No tasks loaded, click 'List Tasks' first\n", "status")
        return

      win = tk.Toplevel(root)
      win.title("Train Policy")
      win.configure(bg=BG)
      win.geometry("800x700")
      win.resizable(False, True)
      win.minsize(800, 600)

      # Experiment name
      tk.Label(win, text="Experiment Name:", font=label_font, bg=BG, fg=MUTED).pack(anchor="w", padx=20, pady=(20,6))
      exp_entry = tk.Entry(win, font=label_font)
      exp_entry.pack(fill="x", padx=20, pady=(0,16))

      # Custom job name
      tk.Label(win, text="Custom Job Name:", font=label_font, bg=BG, fg=MUTED).pack(anchor="w", padx=20, pady=(0,6))
      job_entry = tk.Entry(win, font=label_font)
      job_entry.pack(fill="x", padx=20, pady=(0,16))

      # Environment dropdown
      tk.Label(win, text="Select Environment:", font=label_font, bg=BG, fg=MUTED).pack(anchor="w", padx=20, pady=(0,6))
      selected_task = tk.StringVar(value=tasks[0] if tasks else "")
      dropdown_task = tk.OptionMenu(win, selected_task, *tasks)
      dropdown_task.pack(fill="x", padx=20, pady=(0,16))

      # Options label + Load Default button on the same row
      options_header = tk.Frame(win, bg=BG)
      options_header.pack(fill="x", padx=20, pady=(0,6))
      tk.Label(options_header, text="Options:", font=label_font, bg=BG, fg=MUTED).pack(side="left")

      def run_load_default():
        task = selected_task.get()
        if not task:
          append_to_console(consoles[selected_console.get()], "✗ No task selected\n", "status")
          return
        try:
          env_cfg = load_env_cfg(task)
        except Exception as e:
          append_to_console(consoles[selected_console.get()], f"✗ Failed to load default cfg for '{task}': {e}\n", "status")
          return

        lines = extract_default_reward_lines(env_cfg)
        if not lines:
          append_to_console(consoles[selected_console.get()], f"✗ No default rewards found for '{task}'\n", "status")
          return

        options_text.delete("1.0", tk.END)
        options_text.insert("1.0", "\n".join(lines))
        append_to_console(
          consoles[selected_console.get()],
          f"✓ Loaded {len(lines)} default reward option(s) for '{task}'\n",
          "status",
        )

      load_default_btn = tk.Button(
        options_header, text="Load Default", font=btn_font,
        bg=ACCENT2, fg=BG, activebackground=TEXT, activeforeground=BG,
        relief="flat", cursor="hand2", bd=0, padx=10, pady=4,
        command=run_load_default,
      )
      load_default_btn.pack(side="right")
      load_default_btn.bind("<Enter>", lambda e: load_default_btn.config(bg=TEXT))
      load_default_btn.bind("<Leave>", lambda e: load_default_btn.config(bg=ACCENT2))

      # Frame to hold Text + scrollbar
      text_frame = tk.Frame(win)
      text_frame.pack(fill="both", padx=20, pady=(0,16), expand=True)

      options_text = tk.Text(text_frame, height=5, font=label_font, wrap=tk.WORD)
      options_text.pack(side="left", fill="both", expand=True)

      scrollbar = tk.Scrollbar(text_frame, command=options_text.yview)
      scrollbar.pack(side="right", fill="y")

      options_text.config(yscrollcommand=scrollbar.set)

      # Description label
      tk.Label(win, text="Description:", font=label_font, bg=BG, fg=MUTED).pack(anchor="w", padx=20, pady=(0,6))
      # Frame to hold Text + scrollbar
      desc_text_frame = tk.Frame(win)
      desc_text_frame.pack(fill="both", padx=20, pady=(0,16), expand=True)

      desc_text = tk.Text(desc_text_frame, height=5, font=label_font, wrap=tk.WORD)
      desc_text.pack(side="left", fill="both", expand=True)

      desc_scrollbar = tk.Scrollbar(desc_text_frame, command=desc_text.yview)
      desc_scrollbar.pack(side="right", fill="y")

      desc_text.config(yscrollcommand=desc_scrollbar.set)

      def confirm():
        experiment_name = exp_entry.get().strip()
        custom_job_name = job_entry.get().strip()
        environment_name = selected_task.get()
        if not (experiment_name and custom_job_name and environment_name):
          append_to_console(consoles[selected_console.get()], "✗ All three fields are required\n", "status")
          return

        lines = options_text.get("1.0", tk.END).splitlines()
        extra_opts = " ".join(f"--{line.strip()}" for line in lines if line.strip())
        description = desc_text.get("1.0", tk.END)

        label = f"train:{custom_job_name}:{environment_name}"
        cmd = f'tsp uv run train {environment_name} --agent.run-name {custom_job_name} {extra_opts}'
        save_training_snapshot(environment_name, custom_job_name, extra_opts, description)

        win.destroy()
        launch_process(cmd, consoles[selected_console.get()], label=label)

      btn = tk.Button(win, text="▶  Start Training", font=btn_font,
                      bg=ACCENT, fg=BG, relief="flat", cursor="hand2",
                      padx=12, pady=8, command=confirm)
      btn.pack(padx=20, pady=(0,20))

    def run_sync_checkpoints():
      user, ip, remote_path, setup_flag = remote_setup
      if setup_flag != 1 :
        append_to_console(consoles[selected_console.get()], f'[ERROR] Remote setup has not been done !\n', "status")
        return

      win = tk.Toplevel(root)
      win.title("Sync Checkpoints (rsync)")
      win.configure(bg=BG)
      win.geometry("520x480")
      win.resizable(False, False)

      # Filter row
      filter_frame = tk.Frame(win, bg=BG)
      filter_frame.pack(fill="x", padx=20, pady=(16, 0))

      filter_enabled = tk.BooleanVar(value=False)
      filter_checkbox = tk.Checkbutton(
          filter_frame,
          text="Filter by run name ({run_date}_{run_name})",
          variable=filter_enabled,
          onvalue=True,
          offvalue=False,
          font=label_font,
          bg=BG,
          fg=TEXT,
          activebackground=BG,
          activeforeground=ACCENT,
          selectcolor=BG,
          relief="flat",
          anchor="w",
          padx=0,
          pady=2,
          bd=0,
          highlightthickness=0,
      )
      filter_checkbox.pack(anchor="w")

      tk.Label(win, text="Run name filter (substring match):", font=label_font,
               bg=BG, fg=MUTED).pack(anchor="w", padx=20, pady=(10, 6))
      filter_entry = tk.Entry(win, font=label_font)
      filter_entry.pack(fill="x", padx=20)

      status_label = tk.Label(win, text="", font=label_font, bg=BG, fg=MUTED, wraplength=470, justify="left")
      status_label.pack(anchor="w", padx=20, pady=(10, 0))

      def confirm():

        use_filter = filter_enabled.get()
        pattern = filter_entry.get().strip()

        if use_filter and not pattern:
          status_label.config(
            text="✗ Filter is enabled but empty",
            fg=DANGER
          )
          return

        # Ask for SSH password
        password = simpledialog.askstring(
          "SSH Password",
          f"Enter SSH password for {user}:",
          show="*",
          parent=win,
        )

        if password is None:
          return

        LOG_ROOT.mkdir(parents=True, exist_ok=True)

        remote_source = f"{user}@{ip}:{remote_path}/logs/"
        local_dest = f"{LOG_ROOT}/"

        # SSH options
        ssh_cmd = "ssh -o StrictHostKeyChecking=no"

        extra_env = {
          "SSHPASS": password
        }

        rsh = f"sshpass -e {ssh_cmd}"

        rsync_flags = f"-avzP -e \"{rsh}\""

        if use_filter:
          rsync_flags += (
            f" --include='*/'"
            f" --include='*{pattern}*/**'"
            f" --exclude='*'"
          )

        cmd = f"rsync {rsync_flags} {remote_source} {local_dest}"

        win.destroy()

        launch_process(
          cmd,
          consoles[selected_console.get()],
          label=f"sync:{user}@{ip}",
          extra_env=extra_env,
        )

      btn = tk.Button(win, text="⇄  Sync Checkpoints", font=btn_font,
                      bg=ACCENT, fg=BG,
                      relief="flat", cursor="hand2",
                      padx=12, pady=8,
                      command=confirm)
      btn.pack(padx=20, pady=(20, 10))

    def setup_remote():
      win = tk.Toplevel(root)
      win.title("Setup remote repo")
      win.configure(bg=BG)
      win.geometry("520x480")
      win.resizable(False, False)

      def add_label(text):
        tk.Label(win, text=text, font=label_font, bg=BG, fg=MUTED).pack(anchor="w", padx=20, pady=(14, 6))

      def add_entry(show=None):
        e = tk.Entry(win, font=label_font, show=show) if show else tk.Entry(win, font=label_font)
        e.pack(fill="x", padx=20)
        return e

      add_label("Remote user:")
      user_entry = add_entry()

      add_label("Remote IP / hostname:")
      ip_entry = add_entry()

      add_label("Remote path (e.g. /home/user/pal_mjlab/):")
      path_entry = add_entry()

      status_label = tk.Label(win, text="", font=label_font, bg=BG, fg=MUTED, wraplength=470, justify="left")
      status_label.pack(anchor="w", padx=20, pady=(10, 0))

      def confirm():
        user = user_entry.get().strip()
        ip = ip_entry.get().strip()
        remote_path = path_entry.get().strip().rstrip("/")

        if not (user and ip and remote_path):
          status_label.config(text="✗ User, IP and remote path are required", fg=DANGER)
          return
        
        cmd = f"[ -e '{remote_path}' ] && echo 'True' || echo 'False'"

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        password = simpledialog.askstring(
            "SSH Password",
            "Enter SSH password:",
            show="*",
            parent=win,
        )

        if password is None:
            return

        ssh.connect(
          hostname=ip,
          username=user,
          password=password,
          allow_agent=False,
          look_for_keys=False,
        )

        stdin, stdout, stderr = ssh.exec_command(cmd)

        if stdout.read().decode() == "False\n" :
          append_to_console(consoles[selected_console.get()], f'[WARNING] Directory {remote_path} does not exist in the remote PC !\n', "status")
        else :
          write_remote_setup(user, ip, remote_path)
          remote_setup = user, ip, remote_path, 1
          append_to_console(consoles[selected_console.get()], f'Remote directory {remote_path} set\n', "status")

        ssh.close()

        win.destroy()

      btn = tk.Button(win, text="Confirm", font=btn_font,
                      bg=ACCENT, fg=BG,
                      relief="flat", cursor="hand2",
                      padx=12, pady=8,
                      command=confirm)
      btn.pack(padx=20, pady=(20, 10))

    def clear_selected_terminal ():
      consoles[selected_console.get()].delete("1.0", tk.END)
      append_to_console(consoles[selected_console.get()], f'Terminal {selected_console.get() + 1} ready.\n\n', "status")

    def refresh_process_list():
      proc_listbox.delete(0, tk.END)
      for label in processes.keys():
        proc_listbox.insert(tk.END, label)

    def terminate_selected_process():
      selection = proc_listbox.curselection()
      if not selection:
        return
      label = proc_listbox.get(selection[0])
      proc = processes.get(label)
      if proc:
        try:
          kill_proc_tree(proc)
        except Exception as e:
          print(f"Error terminating {label}: {e}")
        processes.pop(label, None)
      refresh_process_list()

    tk.Label(left, text="Select Terminal:", font=label_font, bg=PANEL, fg=MUTED).pack(anchor="w", padx=16, pady=(8,2))

    for i, name in enumerate(terminal_names):
      tk.Radiobutton(
        left,
        text=name,
        variable=selected_console,  # only one selected at a time
        value=i,
        font=label_font,
        bg=PANEL,
        fg=TEXT,
        selectcolor=BG,  # color inside the radio button
        activebackground=PANEL,
        activeforeground=ACCENT,
        highlightthickness=0,
      ).pack(anchor="w", padx=20)

    tk.Frame(left, bg=PANEL, height=8).pack(fill="x", padx=16)
    make_button(left, "Train Policy", ACCENT, run_train)
    make_button(left, "List tasks",  ACCENT2, run_list_envs)
    make_button(left, "▶  Deploy", ACCENT, run_deploy)
    make_button(left, "⇄  Sync Checkpoints", ACCENT2, run_sync_checkpoints)
    make_button(left, "Setup Remote Folder", ACCENT, setup_remote)
    remote_var = tk.BooleanVar(value=False)
    remote_checkbox = tk.Checkbutton(
        left,
        text="Remote",
        variable=remote_var,
        onvalue=True,
        offvalue=False,
        font=label_font,
        bg=BG,
        fg="#e8eaf0",
        activebackground=BG,
        activeforeground="#00d4aa",  
        selectcolor=BG,               
        relief="flat",
        anchor="w",                   
        padx=5,
        pady=2,
        bd=0,
        highlightthickness=0,
    )
    remote_checkbox.pack(fill="x", padx=20, pady=(0,16))

    tk.Frame(left, bg=MUTED, height=1).pack(fill="x", padx=16, pady=(8, 4))
    tk.Label(left, text="Job Queue (tsp):", font=label_font, bg=PANEL, fg=MUTED).pack(anchor="w", padx=16, pady=(0, 2))
    make_button(left, "Check tsp",      ACCENT2, run_tsp)
    make_button(left, "Check tsp -t",   ACCENT2, run_tsp_t)
    make_button(left, "Remove tsp Job", DANGER,  run_tsp_remove)

    tk.Label(left, text="Running Processes:", font=label_font, bg=PANEL, fg=MUTED).pack(anchor="w", padx=16, pady=(16, 4))

    proc_listbox = tk.Listbox(left, font=label_font, bg=BG, fg=TEXT, selectbackground=ACCENT2, height=6)
    proc_listbox.pack(fill="x", padx=16, pady=(0, 8))

    make_button(left, "Terminate Selected", DANGER, terminate_selected_process)

    # Spacer + quit at bottom
    left.pack_propagate(False)
    spacer = tk.Frame(left, bg=PANEL)
    spacer.pack(fill="both", expand=True)

    make_button(left, "Clear terminal", DANGER, clear_selected_terminal)
    make_button(left, "X  Quit", DANGER, cleanup)

    # Right console frame
    right = tk.Frame(root, bg=BG)
    right.grid(row=0, column=1, sticky="nsew", padx=(4, 10), pady=10)

    # Configure 2 rows and 2 columns to expand evenly
    for r in range(2):
      right.rowconfigure(r, weight=1, uniform="row")
    for c in range(2):
      right.columnconfigure(c, weight=1, uniform="col")

    # Create 4 consoles in a 2x2 grid
    consoles = []
    for i in range(4):
      r, c = divmod(i, 2)  # row, column
      c_widget = scrolledtext.ScrolledText(
        right, font=console_font,
        bg=PANEL, fg=TEXT, insertbackground=ACCENT,
        relief="flat", bd=0, padx=12, pady=10,
        wrap=tk.WORD, state="normal"
      )
      c_widget.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
      c_widget.tag_config("cmd",    foreground=ACCENT, font=tkfont.Font(family="Courier New", size=10, weight="bold"))
      c_widget.tag_config("status", foreground=ACCENT2, font=tkfont.Font(family="Courier New", size=10, weight="bold"))
      append_to_console(c_widget, f"Terminal {i+1} ready.\n\n", "status")

      c_widget.bind("<Key>", lambda e: "break" if not (e.state & 0x4 and e.keysym in ("c", "C", "a", "A")) else None)

      c_widget.bind("<Button-1>", lambda e, idx=i: selected_console.set(idx))

      consoles.append(c_widget)

    root.mainloop()

def main():
  return tyro.cli(open_menu, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected. Exiting cleanly...")
    cleanup()