import os
import sys
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsProject

# Import your existing class code
from .cross_section_editor import CrossSectionEditorApp

class CrossSectionEditorPlugin:
    """QGIS Plugin for the Cross Section Editor"""

    def __init__(self, iface):
        """Constructor.
        
        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = "Cross Section Editor"
        self.toolbar = self.iface.addToolBar('Cross Section Editor')
        self.toolbar.setObjectName('CrossSectionEditor')
        
        # Initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)

    def add_action(self, icon_path, text, callback, enabled_flag=True, 
                   add_to_menu=True, add_to_toolbar=True, status_tip=None,
                   whats_this=None, parent=None):
        """Add a toolbar icon to the toolbar.
        
        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str
        
        :param text: Text that should be shown in menu items for this action.
        :type text: str
        
        :param callback: Function to be called when the action is triggered.
        :type callback: function
        
        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool
        
        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool
        
        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool
        
        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str
        
        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.
        :type whats_this: str
        
        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget
        
        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """
        
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)
        
        if status_tip is not None:
            action.setStatusTip(status_tip)
        
        if whats_this is not None:
            action.setWhatsThis(whats_this)
        
        if add_to_toolbar:
            self.toolbar.addAction(action)
        
        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)
        
        self.actions.append(action)
        
        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        
        icon_path = os.path.join(self.plugin_dir, 'icon.png')
        self.add_action(
            icon_path,
            text="Open Cross Section Editor",
            callback=self.run,
            parent=self.iface.mainWindow())

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                "Cross Section Editor",
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    def run(self):
        """Run method that performs all the real work"""
        try:
            # Create and show your dialog. Note: we're using your existing app
            self.editor_app = CrossSectionEditorApp(self.iface)
            self.editor_app.show()
        except Exception as e:
            QMessageBox.critical(
                None, 
                "Cross Section Editor Error",
                f"Error starting the Cross Section Editor: {str(e)}"
            )
            