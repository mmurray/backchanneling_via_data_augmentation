import shelve


def shelve_it(file_name):
    d = shelve.open(file_name, protocol=5)

    def decorator(func):
        def new_func(*args):
            key = repr(tuple(args))
            if key not in d:
                d[key] = func(*args)
            return d[key]

        return new_func

    return decorator


def component_shelf(file_name):
    d = shelve.open(file_name, protocol=5)

    def decorator(func):
        def new_func(path, name, **kwargs):
            normalized_name = name.replace("_s.mp4", "")
            key = repr(tuple(normalized_name)) + repr(sorted(kwargs["components"]))
            if key in d:
                return d[key]
            else:
                for existing_key in d.keys():
                    if not existing_key.startswith(repr(tuple(normalized_name))):
                        continue
                    # Check that it has all necessary components
                    match = True
                    for component in kwargs["components"]:
                        if component not in existing_key:
                            match = False
                            break
                    if match:
                        return d[existing_key]

            d[key] = func(path, name, )
            return d[key]

        return new_func

    return decorator
