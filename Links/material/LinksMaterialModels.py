#
# Links Materials Constitutive Models Module (UNNAMED Program)
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | April 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Import from string
import importlib
#
#                                                         Links material constitutive models
# ==========================================================================================
# Set material procedures for a given Links constitutive model
def LinksMaterialProcedures(model_name):
    # Set mapping between Links constitutive model name and associated procedures module
    # name
    name_dict = {'ELASTIC':'Links_elastic','VON_MISES':'Links_von_mises'}
    # Get Links constitutive model module
    model_module = importlib.import_module('Links.material.models.' + name_dict[model_name])
    # Set Links constitutive model procedures
    setRequiredProperties,init,writeMaterialProperties,linksXPROPS,linksXXXXVA = \
                                                      model_module.setLinksModelProcedures()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return [setRequiredProperties,init,writeMaterialProperties,linksXPROPS,linksXXXXVA]
