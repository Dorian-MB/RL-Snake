"""Utility functions for agent management and operations."""

from ..environment.utils import ModelLoader, ModelRenderer, get_env

# Re-export for backward compatibility
__all__ = ["ModelLoader", "ModelRenderer", "get_env"]

import logging

import torch, sys

def is_directml_available() -> tuple[bool, torch.device | str]:
    """Vérifie si DirectML est disponible et fonctionnel"""
    try:
        import torch_directml
        dml = torch_directml.device()
        # Test simple pour vérifier que le device fonctionne
        x = torch.randn(2, 2, device=dml)
        return True, dml
    except Exception as e:
        return False, str(e)

def is_gpu_available()->tuple[bool, str, torch.device | str]:
    """Vérifie si un GPU est disponible (CUDA ou DirectML)"""
    # Vérifier CUDA 
    if torch.cuda.is_available():
        return True, "cuda", torch.cuda.get_device_name(0)
    
    # Vérifier DirectML 
    dml_available, dml_info = is_directml_available()
    if dml_available:
        return True, "directml", dml_info
    
    return False, "cpu", torch.device("cpu")

def get_system_info() -> None:
    """Affiche des informations détaillées sur le système et les GPUs"""
    print("🖥️  === INFORMATIONS SYSTÈME ===")
    print(f"PyTorch version: {torch.__version__}")
    
    # Informations CPU
    print(f"\n🔧 CPU threads disponibles: {torch.get_num_threads()}")
    
    # Informations CUDA
    print(f"\n🟢 CUDA:")
    print(f"  - Disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - Version CUDA: {torch.version.cuda}")
        print(f"  - Nombre de GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Informations DirectML
    print(f"\n🔵 DirectML:")
    try:
        import torch_directml
        print(f"  - Disponible: ✅")
        
        # Essayer d'obtenir des infos sur le device
        dml_device = torch_directml.device()
        print(f"  - Device: {dml_device}")
        
        # Test de performance simple
        print(f"\n⚡ Test de performance DirectML:")
        import time
        
        # Test DirectML
        start_time = time.time()
        x = torch.randn(10000, 10000, device=dml_device)
        y = torch.randn(10000, 10000, device=dml_device)
        z = torch.mm(x, y)  # Multiplication matricielle
        
        # Pour DirectML, on force la synchronisation en accédant au résultat
        result_sum = z.sum().item()  # Force la synchronisation
        dml_time = time.time() - start_time
        print(f"  - DirectML: {dml_time*1000:.2f}ms")
        
        # Test CPU pour comparaison
        print(f"\n🧪 Benchmark comparatif:")
        start_time = time.time()
        x_cpu = torch.randn(10000, 10000)
        y_cpu = torch.randn(10000, 10000)
        z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_result = z_cpu.sum().item()
        cpu_time = time.time() - start_time
        print(f"  - CPU: {cpu_time*1000:.2f}ms")
        
        # Calcul du speedup
        if dml_time > 0:
            speedup = cpu_time / dml_time
            if speedup > 1:
                print(f"  - 🚀 Accélération DirectML: {speedup:.2f}x plus rapide")
            else:
                print(f"  - 🐌 DirectML: {1/speedup:.2f}x plus lent que CPU")
        
    except ImportError:
        print(f"  - Disponible: ❌ (torch-directml non installé)")
    except Exception as e:
        print(f"  - Erreur: {e}")


class Logger:
    """Logger personnalisé pour l'application.
    Il enregistre les messages dans la console et dans un fichier.
    Les niveaux de log, handeler et formater sont configurables.
    """

    def __init__(self, name="snake_logger", log_file="app.log"):
        self.name = name
        self.log_file = log_file
        self._logger = self.get_logger()

    def get_logger(self):
        """
        Retourne le logger configuré.
        Si le logger n'a pas de handlers, il le configure.
        """
        logger = logging.getLogger(self.name)
        if not logger.hasHandlers():
            return self.setup_handler(logger)
        return logger

    def setup_handler(self, logger):
        # Crée un logger
        logger.setLevel(
            logging.DEBUG
        )  # Niveau global (le plus bas pour capter tous les messages)

        # ----- Handler console -----
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.DEBUG)  # Affiche dans la console

        # ----- Handler fichier -----
        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setLevel(logging.WARNING)  # Enregistre dans le fichier.

        # ----- Format commun -----
        self.set_default_format()

        # ----- Ajout des handlers -----
        logger.addHandler(self.console_handler)
        logger.addHandler(self.file_handler)

        # debug, info, warning, error, critical
        return logger

    def set_default_format(self):
        """Change le format des handlers."""
        self.set_console_format()
        self.set_file_format()

    def set_console_format(
        self, format="\033[1m%(levelname)s\033[0m :\t%(message)s", end="\n"
    ):
        """Change le format du handler console."""
        formatter = logging.Formatter(format + end)
        self.console_handler.setFormatter(formatter)

    def set_file_format(
        self, format="-- %(levelname)s -- %(asctime)s\n%(message)s", end="\n"
    ):
        """Change le format du handler fichier."""
        formatter = logging.Formatter(format + end)
        self.file_handler.setFormatter(formatter)

    def setLevel(self, level):
        self._logger.setLevel(level)
        self.console_handler.setLevel(level)
        self.file_handler.setLevel(level)

    def set_console_level(self, level):
        self.console_handler.setLevel(level)

    def set_file_level(self, level):
        self.file_handler.setLevel(level)

    def close(self):
        for handler in self._logger.handlers:
            handler.close()
            self._logger.removeHandler(handler)

    def __getattr__(self, name):
        return getattr(self._logger, name)
