import threading


class SingletonMeta(type):
    """Metaclass that allows only one instance per scope, so model weights load once.

    The scope is the whole hierarchy under the topmost class that uses this
    metaclass (e.g. one ASRModel of any subclass): constructing again returns
    the existing instance, or raises RuntimeError if the existing instance is
    of an unrelated subclass. A hierarchy whose root sets
    `_singleton_per_class = True` gets one instance per concrete class
    instead (e.g. an LLM extractor and an aggregator side by side).

    Using a metaclass (rather than __new__) means __init__ runs only once,
    so model weights are not reloaded on repeated construction.
    """

    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if getattr(cls, "_singleton_per_class", False):
            scope = cls
        else:  # the topmost class in the MRO built by this metaclass
            scope = [c for c in cls.__mro__ if isinstance(c, SingletonMeta)][-1]
        with SingletonMeta._lock:
            instance = SingletonMeta._instances.get(scope)
            if instance is None:
                instance = SingletonMeta._instances[scope] = super().__call__(*args, **kwargs)
            elif not isinstance(instance, cls):
                raise RuntimeError(
                    f"A {type(instance).__name__} instance already exists; cannot create {cls.__name__}"
                )
        return instance
