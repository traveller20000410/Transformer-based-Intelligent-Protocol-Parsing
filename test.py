from importlib.metadata import metadata

md = metadata('xformers')
for req in md.get_all('Requires-Dist'):
    print(req)

