"""Script to launch a menu to start training or deploy policies"""
"""
Idea :

To have an interactive menu that makes it much easier to laucnh simulation and deploy policies without running and/or modifying bash commands
If used properly, could also become a tool for comparing trained policies quantitatively
"""
import os
import sys
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import tyro
from prettytable import PrettyTable

import threading
import tkinter as tk
from tkinter import scrolledtext
import tkinter.font as tkfont
import re

import mjlab
import mjlab.tasks  # noqa: F401


from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


LOG_ROOT = Path("/home/manuelactis/hpc_remote_logs").resolve()

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

SNAPSHOTS_DIR = Path("../training_snapshots").resolve()

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


tasks = []

# --- Process registry for cleanup ---
processes = {}

def kill_proc_tree(proc):
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
    def launch_process(command, menu_console, label="process", on_complete=None):
      def run():
        proc = subprocess.Popen(
          command,
          shell=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
          executable="/bin/bash",
        )
        processes[label] = proc

        output = []
        menu_console.after(0, append_to_console, menu_console, f"$ {command}\n", "cmd")
        menu_console.after(0, refresh_process_list)

        for line in proc.stdout:
          output.append(line)
          menu_console.after(0, append_to_console, menu_console, line)
        proc.wait()

        status = "\n✓ done" if proc.returncode == 0 else f"\n✗ exited {proc.returncode}"
        menu_console.after(0, append_to_console, menu_console, f"{status}\n\n", "status")
        processes.pop(label, None)
        menu_console.after(0, refresh_process_list)

        if on_complete:
          menu_console.after(0, on_complete, "".join(output))

      thread = threading.Thread(target=run, daemon=True)
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

    tk.Label(left, text="⬡ MJLAB", font=title_font,
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

    def run_ls():
      launch_process("ls -lah", consoles[selected_console.get()], label="ls")

    def run_list_envs():
      def on_complete(output):
        global tasks
        tasks = re.findall(r'\|\s*\d+\s*\|\s*([\w-]+)\s*\|', output)

      launch_process("uv run list_envs", consoles[selected_console.get()], label="list_envs", on_complete=on_complete)

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

    def syncronize():
      launch_process(
          f"cd .. ; hpc tensorboard --checkpoints mn5",
          consoles[selected_console.get()],
          label="Sync"
        )

    def launch_hpc_training(experiment_name, environment_name, custom_job_name, extra_opts, menu_console, description = ""):
      """
      Launches the HPC training workflow sequentially using launch_process:
        1) Build SIF
        2) Deploy SIF
        3) Schedule training job on MN5
      """
      commands = [
        f"cd .. ; hpc job build /home/manuelactis/pal_mjlab -o /home/manuelactis/{experiment_name}.sif",
        f"hpc deploy mn5 /home/manuelactis/{experiment_name}.sif",
        f'hpc job schedule --name {custom_job_name} mn5 {experiment_name}.sif "python -m mjlab.scripts.train {environment_name} --env.scene.num-envs 4096 --agent.run-name {custom_job_name} --agent.logger tensorboard --agent.save-interval 500 {extra_opts}"'
      ]

      save_training_snapshot(environment_name, custom_job_name, extra_opts, description)

      def run_next(index=0):
        if index >= len(commands):
          # all done
          menu_console.after(0, append_to_console, menu_console, "All HPC steps finished.\n", "status")
          menu_console.after(0, menu_console.see, tk.END)
          return

        cmd = commands[index]
        label = f"hpc:{index}:{cmd}"  # unique label per step

        # Launch the command
        def on_complete(output):
          # After this command finishes, run the next
          run_next(index + 1)

        launch_process(cmd, menu_console, label=label, on_complete=on_complete)

      # Start the first command
      run_next(0)

    def run_hpc_train():
      if not tasks:
        append_to_console(consoles[selected_console.get()], "✗ No tasks loaded, click 'List Tasks' first\n", "status")
        return

      win = tk.Toplevel(root)
      win.title("HPC Train Policy")
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

      # Options label
      tk.Label(win, text="Options:", font=label_font, bg=BG, fg=MUTED).pack(anchor="w", padx=20, pady=(0,6))
      # Frame to hold Text + scrollbar
      text_frame = tk.Frame(win)
      text_frame.pack(fill="both", padx=20, pady=(0,16), expand=True)

      options_text = tk.Text(text_frame, height=5, font=label_font, wrap=tk.WORD)
      options_text.pack(side="left", fill="both", expand=True)

      scrollbar = tk.Scrollbar(text_frame, command=options_text.yview)
      scrollbar.pack(side="right", fill="y")

      options_text.config(yscrollcommand=scrollbar.set)

      # Description label
      tk.Label(win, text="Decription:", font=label_font, bg=BG, fg=MUTED).pack(anchor="w", padx=20, pady=(0,6))
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
        win.destroy()
        launch_hpc_training(experiment_name, environment_name, custom_job_name, extra_opts, consoles[selected_console.get()], description)

      btn = tk.Button(win, text="▶  Start HPC Training", font=btn_font,
                      bg=ACCENT, fg=BG, relief="flat", cursor="hand2",
                      padx=12, pady=8, command=confirm)
      btn.pack(padx=20, pady=(0,20))

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

      # Options label
      tk.Label(win, text="Options:", font=label_font, bg=BG, fg=MUTED).pack(anchor="w", padx=20, pady=(0,6))
      # Frame to hold Text + scrollbar
      text_frame = tk.Frame(win)
      text_frame.pack(fill="both", padx=20, pady=(0,16), expand=True)

      options_text = tk.Text(text_frame, height=5, font=label_font, wrap=tk.WORD)
      options_text.pack(side="left", fill="both", expand=True)

      scrollbar = tk.Scrollbar(text_frame, command=options_text.yview)
      scrollbar.pack(side="right", fill="y")

      options_text.config(yscrollcommand=scrollbar.set)

      check_var = tk.BooleanVar(value=False)  # False = unchecked, True = checked

      tk.Label(win, text="Local :", font=label_font, bg=BG, fg=MUTED).pack(anchor="w", padx=20, pady=(0,6))
      checkbox = tk.Checkbutton(
          win,
          text="Local training",
          variable=check_var,
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

      # Description label
      tk.Label(win, text="Decription:", font=label_font, bg=BG, fg=MUTED).pack(anchor="w", padx=20, pady=(0,6))
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

        if (check_var.get()):
          label = f"local_train:{custom_job_name}:{environment_name}"
          cmd = f'uv run train {environment_name} {extra_opts}'
        else :
          label = f"hpc_train:{custom_job_name}:{environment_name}"
          cmd = f'hpc job schedule --name {custom_job_name} mn5 {experiment_name}.sif "python -m mjlab.scripts.train {environment_name} --env.scene.num-envs 4096 --agent.logger tensorboard --agent.run-name {custom_job_name} --agent.save-interval 500 {extra_opts}"'
          save_training_snapshot(environment_name, custom_job_name, extra_opts, description)
        win.destroy()
        
        launch_process(cmd, consoles[selected_console.get()], label=label)

      btn = tk.Button(win, text="▶  Start HPC Training", font=btn_font,
                      bg=ACCENT, fg=BG, relief="flat", cursor="hand2",
                      padx=12, pady=8, command=confirm)
      btn.pack(padx=20, pady=(0,20))

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
    make_button(left, "List Files",    ACCENT,  run_ls)
    make_button(left, "Build - Train Policy", ACCENT2, run_hpc_train)
    make_button(left, "Prebuilt - Train Policy", ACCENT, run_train)
    make_button(left, "List tasks",  ACCENT2, run_list_envs)
    make_button(left, "▶  Deploy", ACCENT, run_deploy)
    make_button(left, "MN Tensorboard", ACCENT2, syncronize)


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
