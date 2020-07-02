"""Compiler handling module."""


#                                                                                   Compiler
# ==========================================================================================
class Compiler:
    """Base class of compiler-related functions."""

    #                                                                    wasThisCompilerUsed
    # --------------------------------------------------------------------------------------
    @staticmethod
    def wasThisCompilerUsed(lib):
        """Check if the library was compiled with a given compiler."""
        raise RuntimeError("This method cannot be called on the base class!")

    #                                                                             symbolName
    # --------------------------------------------------------------------------------------
    @staticmethod
    def symbolName(name):
        """Return the symbol name with a given Fortran name."""
        raise RuntimeError("This method cannot be called on the base class!")

    #                                                                       moduleSymbolName
    # --------------------------------------------------------------------------------------
    @staticmethod
    def moduleSymbolName(module, name):
        """Return the symbol name with a given Fortran name within a module."""
        raise RuntimeError("This method cannot be called on the base class!")

    #                                                                                 symbol
    # --------------------------------------------------------------------------------------
    @classmethod
    def symbol(cls, lib, name):
        """Return the symbol with a given Fortran name."""
        parsedName = cls.symbolName(name)
        try:
            return getattr(lib, parsedName)
        except AttributeError:
            raise RuntimeError("Failed to find symbol " + name)

    #                                                                           moduleSymbol
    # --------------------------------------------------------------------------------------
    @classmethod
    def moduleSymbol(cls, lib, module, name):
        """Return the symbol with a given Fortran name within a module."""
        parsedName = cls.moduleSymbolName(module, name)
        try:
            return getattr(lib, parsedName)
        except AttributeError:
            raise RuntimeError("Failed to find symbol " + name + " in module " + module)


#                                                                                      Ifort
# ==========================================================================================
class Ifort(Compiler):
    """Intel Fortran compiler naming handler."""

    #                                                                    wasThisCompilerUsed
    # --------------------------------------------------------------------------------------
    @staticmethod
    def wasThisCompilerUsed(lib):
        """Check if the library was compiled with Intel Fortran."""
        testFunctions = ['_intel_fast_memset', '_intel_fast_memcpy']
        for testFunction in testFunctions:
            try:
                getattr(lib, testFunction)
                return True
            except AttributeError:
                pass
        return False

    #                                                                             symbolName
    # --------------------------------------------------------------------------------------
    @staticmethod
    def symbolName(name):
        """Return the symbol name with a given Fortran name."""
        return name.lower() + "_"

    #                                                                       moduleSymbolName
    # --------------------------------------------------------------------------------------
    @staticmethod
    def moduleSymbolName(module, name):
        """Return the symbol name with a given Fortran name within a module."""
        return module.lower() + "_mp_" + name.lower() + "_"


#                                                                                   Gfortran
# ==========================================================================================
class Gfortran(Compiler):
    """GNU Fortran compiler naming handler."""

    #                                                                    wasThisCompilerUsed
    # --------------------------------------------------------------------------------------
    @staticmethod
    def wasThisCompilerUsed(lib):
        """Check if the library was compiled with GNU Fortran."""
        testFunctions = ['_gfortran_os_error', '_gfortran_st_write']
        for testFunction in testFunctions:
            try:
                getattr(lib, testFunction)
                return True
            except AttributeError:
                pass
        return False

    #                                                                             symbolName
    # --------------------------------------------------------------------------------------
    @staticmethod
    def symbolName(name):
        """Return the symbol name with a given Fortran name."""
        return name.lower() + "_"

    #                                                                       moduleSymbolName
    # --------------------------------------------------------------------------------------
    @staticmethod
    def moduleSymbolName(module, name):
        """Return the symbol name with a given Fortran name within a module."""
        return "__" + module.lower() + "_MOD_" + name.lower()
