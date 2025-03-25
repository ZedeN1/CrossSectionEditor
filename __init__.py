def classFactory(iface):
    """Load CrossSectionEditor class from file plugin.py
    
    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    from .plugin import CrossSectionEditorPlugin
    return CrossSectionEditorPlugin(iface)
