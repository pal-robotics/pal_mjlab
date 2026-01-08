import os
import requests
import yaml
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class ModelManager:
    def __init__(self, config_file: str = "config.yaml", models_dir: str = "mj_models"):
        """
        Initializes the ModelManager.
        
        Args:
            config_file: Name of the config file (expected in the same folder as this script).
            models_dir: Name of the folder where models will be saved (created next to this script).
        """
        # Determine the directory where this script resides
        self._base_path = Path(__file__).resolve().parent
        
        # Set paths relative to the script location
        self.models_path = self._base_path / models_dir
        self.config_path = self._base_path / config_file
        
        self.config = self._load_config()

    def _load_config(self):
        """Loads the configuration YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def _download_file(url: str, local_path: Path) -> bool:
        """Static helper to download a single file."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return True
        except Exception as e:
            return f"Error downloading {local_path.name}: {e}"

    def download_model(self, model_name: str, force_overwrite: bool = False, max_workers: int = 16) -> bool:
        """
        Downloads the model from MuJoCo Menagerie.
        Returns True if successful (or already exists), False on failure.
        """
        defaults = self.config.get('defaults', {})
        model_conf = self.config.get('models', {}).get(model_name, {})
        
        owner = model_conf.get('owner', defaults.get('owner', 'google-deepmind'))
        repo = model_conf.get('repo', defaults.get('repo', 'mujoco_menagerie'))
        ref = model_conf.get('ref', defaults.get('ref', 'main'))

        target_dir = self.models_path / model_name

        # --- Check Existence ---
        if target_dir.exists():
            if not force_overwrite:
                return True
            else:
                print(f"Warning: Overwriting existing model '{model_name}'...")

        print(f"Checking GitHub for: {model_name} (ref: {ref})")

        # --- Get File Tree ---
        api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
        try:
            r = requests.get(api_url)
            if r.status_code == 404:
                print(f"Error: Repository or Ref '{ref}' not found.")
                return False
            r.raise_for_status()
            tree_data = r.json()
        except Exception as e:
            print(f"Error: API Connection Failed: {e}")
            return False

        # --- Filter Files ---
        target_files = [
            item for item in tree_data.get('tree', [])
            if item['path'].startswith(f"{model_name}/") and item['type'] == 'blob'
        ]

        if not target_files:
            print(f"Error: Model folder '{model_name}' not found in repo.")
            return False

        print(f"Downloading {len(target_files)} files to '{target_dir}'...")

        # --- Parallel Download ---
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for item in target_files:
                repo_path = item['path']
                relative_path = repo_path.replace(f"{model_name}/", "", 1)
                local_file_path = target_dir / relative_path
                
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{repo_path}"
                futures.append(executor.submit(self._download_file, raw_url, local_file_path))

            # Progress Bar
            with tqdm(total=len(futures), unit='file', desc=f"Downloading {model_name}") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not True:
                        print(result) # Print error
                    pbar.update(1)

        print(f"Download complete.")
        return True

    def load_scene(self, model_name: str, scene_name: str) -> str:
        """
        Ensures the model is downloaded and returns the absolute path to the scene XML.
        
        Args:
            model_name: The name of the robot/model folder.
            scene_name: The XML file name (e.g., 'scene.xml' or 'unitree_g1.xml').
            
        Returns:
            str: Absolute path to the XML file.
            
        Raises:
            FileNotFoundError: If the model cannot be downloaded or the scene file is missing.
        """
        # 1. Ensure model is downloaded
        success = self.download_model(model_name)
        if not success:
            raise FileNotFoundError(f"Failed to retrieve model '{model_name}' from repository.")

        # 2. Construct path
        scene_path = self.models_path / model_name / scene_name

        # 3. Verify specific file exists
        if not scene_path.exists():
            # Sometimes users guess the name wrong, let's list available xmls to be helpful
            available = list((self.models_path / model_name).glob("*.xml"))
            available_names = [f.name for f in available]
            raise FileNotFoundError(
                f"Scene file '{scene_name}' not found in model folder.\n"
                f"Available XMLs: {available_names}"
            )

        # 4. Return absolute string path (MuJoCo prefers strings)
        return str(scene_path.resolve())

    def get_model_path(self, model_name: str) -> str:
        """
        Returns the absolute path to the model folder.
        
        Args:
            model_name: The name of the robot/model folder.
            
        Returns:
            str: Absolute path to the model folder.
            
        Raises:
            FileNotFoundError: If the model folder does not exist.
        """
        model_path = self.models_path / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model folder '{model_name}' does not exist at {model_path}.")
        return str(model_path.resolve())