import traceback

try:
    import torch_directml
    print('torch_directml available')
    print('device()', torch_directml.device())
    print('attrs =', sorted([a for a in dir(torch_directml) if not a.startswith('_')]))
    if hasattr(torch_directml, 'device_count'):
        print('device_count()', torch_directml.device_count())
    if hasattr(torch_directml, 'device_name'):
        try:
            print('device_name(0)', torch_directml.device_name(0))
        except Exception:
            traceback.print_exc()
    if hasattr(torch_directml, 'get_device'):
        print('get_device()', torch_directml.get_device())
    if hasattr(torch_directml, 'device_properties'):
        print('device_properties()', torch_directml.device_properties())
    if hasattr(torch_directml, 'get_available_devices'):
        print('available devices =', torch_directml.get_available_devices())
except Exception:
    traceback.print_exc()
