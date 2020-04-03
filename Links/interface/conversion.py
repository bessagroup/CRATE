"""Argument conversion tools."""
import ctypes
import numpy as np


testTypes = {int: ctypes.c_int,
             float: ctypes.c_double,
             bool: ctypes.c_bool}
#            str: ctypes.c_char} (Strings are broken)
CONVERSIONS = {}


#                                                                                  Reference
# ==========================================================================================
class Reference:
    """Mutable class base for immutable objects."""

    def __init__(self, value):
        self.__value = value
        self.__type = type(value)

    #                                                                                  value
    # --------------------------------------------------------------------------------------
    @property
    def value(self):
        """Get or set value."""
        return self.__value

    @value.setter
    def value(self, val):
        self.__value = val

    #                                                                                   type
    # --------------------------------------------------------------------------------------
    @property
    def type(self):
        """Get variable type."""
        return self.__type


#                                                                             ConversionBase
# ==========================================================================================
class ConversionBase(object):
    """Base class for conversion."""

    _value = None

    #                                                                            Constructor
    # --------------------------------------------------------------------------------------
    def __init__(self, value):
        self._value = value

    # #                                                                      __init_subclass__
    # # --------------------------------------------------------------------------------------
    # def __init_subclass__(cls, handles=None, **kwargs):
    #     """Add a subclass to the CONVERSIONS dictionary."""
    #     super().__init_subclass__(**kwargs)
    #     if handles is None:
    #         raise ValueError("Must pass a type handler!")
    #     else:
    #         cls.CONVERSIONS[handles] = cls

    #                                                                           getConverter
    # --------------------------------------------------------------------------------------
    @classmethod
    def getConverter(cls, value):
        """Return a converter class instance based on a value."""
        if type(value) in CONVERSIONS:
            return CONVERSIONS[type(value)](value)
        else:
            raise ValueError("Type of the value not convertible!")

    #                                                                       convertReference
    # --------------------------------------------------------------------------------------
    def convertReference(self, value=None):
        """Get a reference to the variable."""
        if value is None:
            value = self.convertValue()
        return ctypes.pointer(value)

    #                                                                        postCallRestore
    # --------------------------------------------------------------------------------------
    def postCallRestore(self):
        """Restore values post-call."""
        pass

    #                                                                           restoreValue
    # --------------------------------------------------------------------------------------
    def restoreValue(self, convertedValue):
        """Restore a converted integer to Python."""
        return convertedValue.value


#                                                                              IntConversion
# ==========================================================================================
class IntConversion(ConversionBase):
    """Integer conversion."""

    #                                                                           convertValue
    # --------------------------------------------------------------------------------------
    def convertValue(self):
        """Convert an integer to a value for Fortran."""
        return ctypes.c_int(self._value)


CONVERSIONS[int] = IntConversion


#                                                                            FloatConversion
# ==========================================================================================
class FloatConversion(ConversionBase):
    """Float conversion."""

    #                                                                           convertValue
    # --------------------------------------------------------------------------------------
    def convertValue(self):
        """Convert a float to a value for Fortran."""
        return ctypes.c_double(self._value)


CONVERSIONS[float] = FloatConversion


#                                                                             BoolConversion
# ==========================================================================================
class BoolConversion(ConversionBase):
    """Bool conversion."""

    #                                                                           convertValue
    # --------------------------------------------------------------------------------------
    def convertValue(self):
        """Convert a bool to a value for Fortran."""
        return ctypes.c_bool(self._value)


CONVERSIONS[bool] = BoolConversion


#                                                                           StringConversion
# ==========================================================================================
# class StringConversion(ConversionBase, handles=str):
#     """String conversion."""
#
#                                                                               convertValue
#     --------------------------------------------------------------------------------------
#     def convertValue(self):
#         """Convert a float to a value for Fortran."""
#         return ctypes.c_char(self._value)
#

#                                                                            NumpyConversion
# ==========================================================================================
class NumpyConversion(ConversionBase):
    """Numpy array conversion."""

    #                                                                           convertValue
    # --------------------------------------------------------------------------------------
    def convertValue(self):
        """Convert a numpy array to a value for Fortran."""
        self._shape = self._value.shape
        self._convertedValue = self._value.flatten('F')
        return self._convertedValue

    #                                                                       convertReference
    # --------------------------------------------------------------------------------------
    def convertReference(self, value=None):
        """Get a reference to the variable."""
        if value is None:
            value = self.convertValue()
        return np.ctypeslib.as_ctypes(value)

    #                                                                           restoreValue
    # --------------------------------------------------------------------------------------
    def restoreValue(self, convertedValue):
        """Restore a converted array to Python."""
        return np.reshape(convertedValue, self._shape, order='F')


CONVERSIONS[np.ndarray] = NumpyConversion


#                                                                         ReferenceConverter
# ==========================================================================================
class ReferenceConverter(ConversionBase):
    """Reference conversion."""

    #                                                                           convertValue
    # --------------------------------------------------------------------------------------
    def convertValue(self):
        """Convert a Reference to a value for Fortran."""
        self._converter = ConversionBase.getConverter(self._value.value)
        self._convertedValue = self._converter.convertValue()
        return self._convertedValue

    #                                                                       convertReference
    # --------------------------------------------------------------------------------------
    def convertReference(self, value=None):
        """Get a reference to the variable."""
        if value is None:
            value = self.convertValue()
        return self._converter.convertReference(value)

    #                                                                        postCallRestore
    # --------------------------------------------------------------------------------------
    def postCallRestore(self):
        """Restore values post-call."""
        restoredValue = self._converter.restoreValue(self._convertedValue)
        self._value.value = restoredValue

CONVERSIONS[Reference] = ReferenceConverter
