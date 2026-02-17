import sys
import argparse
from omegaconf import OmegaConf, DictConfig

# -----------------------------------------------------------------------------
# Define Custom Aliases Here
# -----------------------------------------------------------------------------
# Format: "flag": {"key": "hydra.config.path", "type": type, "help": "msg", "action": "store_true"}
ALIAS_MAPPING = {
    "--device": {
        "key": "hardware.device",
        "type": str,
        "help": "Chooses the device for PyTorch to use, e.g. 'cuda' or 'cpu'."
    },
    "--lrate": {
        "key": "training.lrate",
        "type": float,
        "help": "Learning rate for training."
    },
    "--display-loss": {
        "key": "misc.display_loss",
        "action": "store_true",
        "help": "Displays the loss and more detailed info as plots at the end of training."
    },
    "--epochs": {
        "key": "training.epochs",
        "type": int,
        "help": "Number of training epochs to run for."
    },
    "--batch-size": {
        "key": "training.batch_size",
        "type": int,
        "help": "Batch size for training."
    },
    "--train-test-split": {
        "key": "training.test_split",
        "type": float,
        "help": "Ratio of images to place into the training set; remainder is placed into the hold out set."
    }
    # Add more aliases as needed
}

class ConfigHandler:
    def __init__(self):
        self._cfg = None
        self._custom_updates = {}

    def parse_custom_args(self):
        """
        Parses custom flags from sys.argv, checks for conflicts with Hydra overrides,
        and removes the custom flags from sys.argv so Hydra doesn't crash.
        """
        parser = argparse.ArgumentParser(add_help=False)  # No -h, let Hydra handle help

        # Dynamically build the parser from ALIAS_MAPPING
        for flag, meta in ALIAS_MAPPING.items():
            kwargs = {"dest": meta["key"]}
            if "type" in meta: kwargs["type"] = meta["type"]
            if "action" in meta: kwargs["action"] = meta["action"]
            parser.add_argument(flag, **kwargs)

        # Parse known custom args, leaving the rest (Hydra overrides) in hydra_args
        args, hydra_args = parser.parse_known_args()

        # --- CONFLICT DETECTION LOGIC ---
        # Look for Hydra overrides in the 'hydra_args' list (e.g. "training.epochs=150")
        hydra_overrides = {
            arg.split("=")[0]: arg.split("=")[1]
            for arg in hydra_args if "=" in arg
        }

        for flag, meta in ALIAS_MAPPING.items():
            key = meta["key"]
            val = getattr(args, key, None)

            # Check if user actually used the flag (ignoring defaults)
            # For store_true, False is default. For others, None is default.
            is_flag_set = val is not None and val is not False

            if is_flag_set:
                # Check if the SAME key is being set via standard Hydra syntax
                # E.g., if "--epochs" is used, we check if "training.epochs" exists in the hydra_args
                conflict = key in hydra_overrides

                if conflict:
                    raise ValueError(
                        f"\n[Config Error] Conflict detected!\n"
                        f"You set '{key}' via the flag '{flag}' AND via a direct override.\n"
                        f"Please use only one.\n"
                    )

                # Store valid custom arg to apply later
                self._custom_updates[key] = val

        # --- ARGV CLEANUP ---
        # IMPORTANT: We must reconstruct sys.argv to remove the flags we just parsed.
        # We keep the script name (sys.argv[0]) and the remaining args (Hydra overrides).
        sys.argv = [sys.argv[0]] + hydra_args

    def setup(self, cfg: DictConfig):
        """Initializes the configuration, merges custom args, and locks it."""

        # Merge the custom arguments we extracted earlier
        if self._custom_updates:
            # Convert our dict back to a dotlist format that OmegaConf understands
            # e.g., {'training.epochs': 150} -> ["training.epochs=150"]
            updates = OmegaConf.from_dotlist([f"{k}={v}" for k, v in self._custom_updates.items()])

            # Merge: Overwrite the defaults from the .yaml file with the command line arguments
            cfg = OmegaConf.merge(cfg, updates)

        # Lock the configuration
        OmegaConf.set_readonly(cfg, True)
        self._cfg = cfg

    def __getitem__(self, key):
        if self._cfg is None:
            raise RuntimeError("Config not initialized. Call Config.setup(cfg) first.")
        return OmegaConf.select(self._cfg, key)

    def __setitem__(self, key, value):
        raise RuntimeError("Config is ReadOnly. Use YAML or CLI args.")

    def __getattr__(self, name):
        """Allows access to _cfg as an attribute"""
        if self._cfg is None:
            raise RuntimeError("Config not initialized")
        # Delegate the attribute access to the underlying OmegaConf object
        return getattr(self._cfg, name)

    def __dir__(self):
        """Helps with autocomplete in interactive consoles (e.g., VSCode or PyCharm IDEs)"""
        if self._cfg is None:
            return super().__dir__()
        # Combine class methods with config keys
        return list(self._cfg.keys()) + super().__dir__()

    def __str__(self):
        """Returns the YAML string representation when calling print(Config)"""
        if self._cfg is None:
            return "<ConfigHandler: Uninitialized>"
        # Return the configuration using a YAML layout
        return OmegaConf.to_yaml(self._cfg)

    def __repr__(self):
        """Standard representation for debugging"""
        return self.__str__()

# Global configuration variable
Config = ConfigHandler()