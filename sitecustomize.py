import os, sys, pathlib

root = pathlib.Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# # Try to import fla; if it fails, inject stub
# try:
#     import fla
#     import fla.ops.kda
#     import fla.modules
#     import fla.models.utils
#     import fla.layers.utils
# except (ImportError, ModuleNotFoundError, RuntimeError, TypeError) as e:
#     # FLA not available - inject stub
#     print(f"FLA not available ({e}), using stub implementation for import compatibility", file=sys.stderr)
#
#     # Add project root to path if not already there
#     project_root = os.path.dirname(os.path.abspath(__file__))
#     if project_root not in sys.path:
#         sys.path.insert(0, project_root)
#
#     # Import and inject the stub
#     try:
#         from stubs import fla_stubs
#         from types import ModuleType
#
#         # Create module hierarchy
#         fla_module = ModuleType('fla')
#         fla_module.__package__ = 'fla'
#         fla_module.__path__ = []
#
#         # ops subpackage
#         ops_module = ModuleType('fla.ops')
#         ops_module.__package__ = 'fla.ops'
#         ops_module.__path__ = []
#
#         # ops.kda subpackage
#         kda_module = ModuleType('fla.ops.kda')
#         kda_module.chunk_kda = fla_stubs.chunk_kda
#         kda_module.fused_recurrent_kda = fla_stubs.fused_recurrent_kda
#
#         # ops.kda.gate submodule
#         gate_module = ModuleType('fla.ops.kda.gate')
#         gate_module.fused_kda_gate = fla_stubs.fused_kda_gate
#
#         # modules subpackage
#         modules_module = ModuleType('fla.modules')
#         modules_module.FusedRMSNormGated = fla_stubs.FusedRMSNormGated
#         modules_module.ShortConvolution = fla_stubs.ShortConvolution
#
#         # models.utils subpackage
#         models_module = ModuleType('fla.models')
#         models_utils_module = ModuleType('fla.models.utils')
#         models_utils_module.Cache = fla_stubs.Cache
#
#         # layers.utils subpackage
#         layers_module = ModuleType('fla.layers')
#         layers_utils_module = ModuleType('fla.layers.utils')
#         layers_utils_module.get_unpad_data = fla_stubs.get_unpad_data
#         layers_utils_module.index_first_axis = fla_stubs.index_first_axis
#         layers_utils_module.pad_input = fla_stubs.pad_input
#
#         # Register all modules
#         sys.modules['fla'] = fla_module
#         sys.modules['fla.ops'] = ops_module
#         sys.modules['fla.ops.kda'] = kda_module
#         sys.modules['fla.ops.kda.gate'] = gate_module
#         sys.modules['fla.modules'] = modules_module
#         sys.modules['fla.models'] = models_module
#         sys.modules['fla.models.utils'] = models_utils_module
#         sys.modules['fla.layers'] = layers_module
#         sys.modules['fla.layers.utils'] = layers_utils_module
#
#         print("FLA stub modules registered successfully", file=sys.stderr)
#
#     except ImportError as e:
#         print(f"Warning: Could not import fla_stubs: {e}", file=sys.stderr)