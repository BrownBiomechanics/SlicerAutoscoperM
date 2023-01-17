import slicer
from slicer.ScriptedLoadableModule import *

#
# AutoscoperMStub
#

class AutoscoperMStub(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "AutoscoperM"
    self.parent.categories = ["Tracking"]
    self.parent.dependencies = []
    self.parent.contributors = ["Anthony Lombardi (Kitware), Bardiya Akhbari (Brown University), Amy Morton (Brown University), Beatriz Paniagua (Kitware), Jean-Christophe Fillion-Robin (Kitware)"]
    self.parent.helpText = """
This is a placeholder module to tell the user that AutoscoperM extension is not available on the platform.
See more information in the <a href="https://autoscoperm.slicer.org/">extension documentation</a>.
"""
    self.parent.acknowledgementText = ""

#
# AutoscoperMStubWidget
#

class AutoscoperMStubWidget(ScriptedLoadableModuleWidget):
  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

  def enter(self):
    """
    Called each time the user opens this module.
    """
    slicer.util.messageBox("AutoscoperM is not supported on this platform.<br>"
      "See <a href='https://autoscoperm.slicer.org/'>Slicer Autoscoper website</a> for details.")