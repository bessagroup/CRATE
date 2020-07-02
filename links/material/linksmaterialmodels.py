#
# Links Materials Constitutive Models Module (CRATE Program)
# ==========================================================================================
# Summary:
# Module containing procedures related to the definition of the finite element code Links
# constitutive models.
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
def getlinksmodel(model_name):
    # Set mapping between Links constitutive model name and associated procedures module
    # name
    name_dict = {'ELASTIC': 'links_elastic', 'VON_MISES': 'links_von_mises'}
    # Get Links constitutive model module
    model_module = importlib.import_module('links.material.models.' + name_dict[model_name])
    # Set Links constitutive model procedures
    getrequiredproperties, init, writematproperties, linksxprops, linksxxxxva = \
        model_module.getlinksmodelprocedures()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return [getrequiredproperties, init, writematproperties, linksxprops, linksxxxxva]
