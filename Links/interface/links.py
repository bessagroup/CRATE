"""LINKS module."""
import os
import ctypes
import subprocess
from Links.interface.compiler import Ifort, Gfortran
from Links.interface.conversion import ConversionBase, testTypes


COMPILING_LINKS_MSG = "Compiling Links..."
USING_BRANCH = 'python-interface'
LINKS_LIBRARY_NAME = 'LINKS.so'
TEST_COMPILERS = [Gfortran, Ifort]


#                                                                               callFunction
# ------------------------------------------------------------------------------------------
def callFunction(func, *args, returnType=None):
    """Actual function call."""
    # First make the variables nice and tidy for Fortran (references and ctypes types)
    conversionInstances = [ConversionBase.getConverter(a) for a in args]
    parsedArgs = [a.convertReference() for a in conversionInstances]
    # Now set the return type
    if returnType is not None:
        func.restype = returnType
    # True shit happens here
    value = func(*parsedArgs)
    # Restore values, in case References were passed
    for a in conversionInstances:
        a.postCallRestore()
    return value


#                                                                                     Symbol
# ==========================================================================================
class Symbol:
    """Binary symbol."""

    #                                                                            Constructor
    # --------------------------------------------------------------------------------------
    def __init__(self, name, lib, compiler, actualFunction=None, symbolName=None):
        self.__name = name
        self.__lib = lib
        self.__compiler = compiler
        self.__func = actualFunction
        if symbolName is None:
            self.__symbolName = self.__compiler.symbolName(self.__name)
        else:
            self.__symbolName = symbolName

    #                                                                               __call__
    # --------------------------------------------------------------------------------------
    def __call__(self, *args, returnType=int):
        """Dispatch call to the symbol."""
        # Convert the return type to ctypes
        if returnType not in testTypes:
            raise TypeError("Specified return type not supported!")
        c_returnType = testTypes[returnType]
        # Select how to call the function
        if self.__func is None:
            # Search for the function
            symbol = self.__compiler.symbol(self.__lib, self.__name)
            return callFunction(symbol, *args, returnType=c_returnType)
        else:
            # We already know it, is inside of a module (check __getattr__)
            return callFunction(self.__func, *args, returnType=c_returnType)

    #                                                                            __getattr__
    # --------------------------------------------------------------------------------------
    def __getattr__(self, name):
        """Return symbol inside a module."""
        symbol = self.__compiler.moduleSymbol(self.__lib, self.__name, name)
        symbolName = self.__compiler.moduleSymbolName(self.__name, name)
        newSymbolInstance = Symbol(name, self.__lib, self.__compiler,
                                   actualFunction=symbol, symbolName=symbolName)
        return newSymbolInstance

    #                                                                            __getitem__
    # --------------------------------------------------------------------------------------
    def __getitem__(self, key):
        """Return the value of a symbol with a given type."""
        symbol = self.__getVariableRef(key)
        return symbol.value

    #                                                                            __setitem__
    # --------------------------------------------------------------------------------------
    def __setitem__(self, key, value):
        """Set the value of a symbol with a given type."""
        symbol = self.__getVariableRef(key)
        symbol.value = value

    #                                                                       __getVariableRef
    # --------------------------------------------------------------------------------------
    def __getVariableRef(self, key):
        """Get the reference of a given variable."""
        if key not in testTypes:
            raise TypeError("Can't convert type " + key)
        converter = testTypes[key]
        return converter.in_dll(self.__lib, self.__symbolName)


#                                                                                      Links
# ==========================================================================================
class Links:
    """Interface class for Links."""

    #                                                                            Constructor
    # --------------------------------------------------------------------------------------
    def __init__(self, src=None, bin=None, makeFlags='-Bj optm python'):
        if (src, bin) is (None, None):
            raise ValueError("No Links source or binary path were passed!")
        if bin is not None:
            # A path for the binary was passed. Check if a file or a directory
            if os.path.isdir(bin):
                self.__binFile = os.path.join(bin, LINKS_LIBRARY_NAME)
                if not os.path.isfile(self.__binFile):
                    raise ValueError("Links library not found in bin path passed!")
            elif os.path.isfile(bin):
                self.__binFile = bin
            else:
                raise ValueError("Binary path supplied is neither a file or a directory!")
        else:
            # Only a path for the source directory was passed. Check if correct
            if not os.path.isdir(src):
                raise ValueError("Source path supplied is not a directory!")
            # Check if the src path is valid by finding links.f90 (fragile, but works)
            if not os.path.isfile(os.path.join(src, 'links.f90')):
                raise ValueError("Source path supplied is not Links' source directory!")
            # Check if git is available
            if os.path.isdir(os.path.join(src, '.git')):
                # TODO
                pass
            # If we got here then the directory is probably safe. Compile Links
            print(COMPILING_LINKS_MSG)
            binDir = os.path.join(src, os.pardir, "bin")
            with open(os.path.join(binDir, "links.log"), "w") as logFile:
                status = subprocess.run(['make'] + makeFlags.split(),
                                        cwd=src, stdout=logFile, stderr=logFile)
            if status.returncode != 0:
                raise RuntimeError("Links compilation failed! Check log in bin directory!")
            self.__binFile = os.path.join(binDir, LINKS_LIBRARY_NAME)
            if not os.path.isfile(self.__binFile):
                raise ValueError("Links library not found after compilation!")
        # Time to actually load the library
        self.__links = ctypes.cdll.LoadLibrary(self.__binFile)
        # Find the compiler used
        for compiler in TEST_COMPILERS:
            result = compiler.wasThisCompilerUsed(self.__links)
            if result:
                self.__compiler = compiler
                break

    #                                                                            __getattr__
    # --------------------------------------------------------------------------------------
    def __getattr__(self, name):
        """Actual symbol lookup function."""
        symbol = Symbol(name, self.__links, self.__compiler)
        return symbol
