import importlib, traceback
try:
    m = importlib.import_module('services.cleaner_service')
    print('module file:', getattr(m,'__file__',None))
    print('has DataCleaner:', hasattr(m,'DataCleaner'))
    print('attrs:', [a for a in dir(m) if not a.startswith('_')])
except Exception as e:
    print('IMPORT ERROR:')
    traceback.print_exc()
