DEBUG = False  # Set to True to enable debug logging

# QGIS imports
from qgis.core import (
    QgsVectorLayer,
    QgsProject,
    QgsWkbTypes,
    QgsProcessingFeatureSourceDefinition,
    QgsFeatureRequest,
    QgsExpression,
    QgsExpressionContext,
    QgsExpressionContextUtils,
    QgsRectangle
)
from qgis.analysis import (QgsNativeAlgorithms)

import processing
from processing.core.Processing import Processing

# PyQt imports
from qgis.PyQt.QtCore import (Qt, QAbstractTableModel, QModelIndex, pyqtSignal, QPoint, QTimer)
from qgis.PyQt.QtGui import (QKeySequence, QColor, QBrush, QCursor)
from qgis.PyQt.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QSplitter,
    QTableView, QHeaderView, QPushButton, QFileDialog, QListWidget, QLabel,
    QCheckBox, QComboBox, QLineEdit, QMessageBox, QGridLayout, QGroupBox, QMenu,
    QStatusBar, QDialog, QDialogButtonBox,
    QAbstractItemView, QSizePolicy, QAction
)

# Standard library imports
import os
from io import StringIO
from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
from collections import deque
from ast import literal_eval

# Parses coordinates from POINT (x y) and POINT Z (x y z) WKT strings
_WKT_POINT_RE = re.compile(
    r'POINT\s*(?:Z\s*)?\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)(?:\s+([\d.eE+\-]+))?\s*\)',
    re.IGNORECASE
)

# Third-party imports
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

class EditablePandasModel(QAbstractTableModel):
    dataChanged = pyqtSignal(QModelIndex, QModelIndex)

    def __init__(self, data):
        super().__init__()
        self._data = data
        self.cut_indices = set()  # Track indices that will be cut

    def rowCount(self, index=None):
        return self._data.shape[0]

    def columnCount(self, index=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
                return str(self._data.iloc[index.row(), index.column()])
            elif role == Qt.ItemDataRole.BackgroundRole:
                if index.row() in self.cut_indices:
                    return QBrush(QColor(192, 192, 192))  # Light red for cut rows
            elif role == Qt.ItemDataRole.ForegroundRole:
                if index.row() in self.cut_indices:
                    return QBrush(QColor(0, 0, 0))  # Black text color for cut rows

        return None

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Orientation.Vertical:
                return str(self._data.index[section])
        return None

    def flags(self, index):
        if index.isValid():
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
        return Qt.ItemFlag.NoItemFlags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if index.isValid() and role == Qt.ItemDataRole.EditRole:
            try:
                # Try to convert value to proper type
                original_value = self._data.iloc[index.row(), index.column()]
                if isinstance(original_value, (int, float)):
                    converted_value = float(value)
                else:
                    converted_value = value

                # Explicitly cast to the column's type if needed
                column_dtype = self._data.iloc[:, index.column()].dtype
                if pd.api.types.is_integer_dtype(column_dtype):
                    converted_value = int(converted_value)  # Convert to int if the column is of integer type
                elif pd.api.types.is_float_dtype(column_dtype):
                    converted_value = float(converted_value)  # Convert to float if the column is of float type

                # Set the value in the dataframe
                self._data.iloc[index.row(), index.column()] = converted_value

                # Emit dataChanged with proper indices
                top_left = self.index(index.row(), index.column())
                bottom_right = self.index(index.row(), index.column())
                self.dataChanged.emit(top_left, bottom_right)  # Correct emission with indices
                self.layoutChanged.emit()  # Notify that the layout has changed
                return True
            except (ValueError, TypeError):
                return False
        return False

    def set_cut_indices(self, indices):
        self.cut_indices = set(indices)
        top_left = self.index(0, 0)
        bottom_right = self.index(self.rowCount()-1, self.columnCount()-1)
        self.dataChanged.emit(top_left, bottom_right)
        self.layoutChanged.emit()  # Notify that the layout has changed

    def get_dataframe(self):
        return self._data

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        # Make layout expand with window resizing
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(400, 300)

        layout = QVBoxLayout(self)

        # Help text
        help_label = QLabel(
            "List of expected column names in descending order of preference.\n"
            "Column names are represented as case-insensitive strings (e.g., 'x'), while column indices are integers (0-based).\n"
            "First match will be applied on per file basis."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        # X Column preferences
        x_group = QGroupBox("X Column Preferences")
        x_layout = QVBoxLayout()
        self.x_text = QLineEdit()
        self.x_text.setToolTip(
            "Python list of column names/indices to try as the X (chainage/distance) column,\n"
            "in descending order of preference. Names are case-insensitive strings; indices are\n"
            "0-based integers. The first match found in a file is used.\n\n"
            "Example: ['x', 'x (m)', 'chainage', 'w', 0]"
        )
        x_layout.addWidget(self.x_text)
        x_group.setLayout(x_layout)
        layout.addWidget(x_group)

        # Y Column preferences
        y_group = QGroupBox("Y Column Preferences")
        y_layout = QVBoxLayout()
        self.y_text = QLineEdit()
        self.y_text.setToolTip(
            "Python list of column names/indices to try as the Y (elevation/depth) column,\n"
            "in descending order of preference. Names are case-insensitive strings; indices are\n"
            "0-based integers. The first match found in a file is used.\n\n"
            "Example: ['y', 'z', 'h', 1]"
        )
        y_layout.addWidget(self.y_text)
        y_group.setLayout(y_layout)
        layout.addWidget(y_group)

        # N Column preferences
        n_group = QGroupBox("N (Roughness) Column Preferences. [] if not needed")
        n_layout = QVBoxLayout()
        self.n_text = QLineEdit()
        self.n_text.setToolTip(
            "Python list of column names to try as the Manning's N (roughness) column,\n"
            "in descending order of preference. Use [] if no roughness column is expected.\n\n"
            "Example: ['n', 'm', 'Mannings n']"
        )
        n_layout.addWidget(self.n_text)
        n_group.setLayout(n_layout)
        layout.addWidget(n_group)

        # W Column preferences
        x_unsortable_group = QGroupBox("Unsortable Column Preferences (eg 'W' in HW tables).")
        x_unsortable_layout = QVBoxLayout()
        self.x_unsortable_text = QLineEdit()
        self.x_unsortable_text.setToolTip(
            "Python list of column names whose values cannot be meaningfully sorted as numbers.\n"
            "When a file's X column matches one of these names, row reordering is skipped.\n"
            "Useful for HW (head-water) tables where 'W' is a water level, not a chainage.\n\n"
            "Example: ['w']"
        )
        x_unsortable_layout.addWidget(self.x_unsortable_text)
        x_unsortable_group.setLayout(x_unsortable_layout)
        layout.addWidget(x_unsortable_group)

        # Easting Column preferences
        easting_group = QGroupBox("Easting Column Preferences. (WKT takes preference)")
        easting_layout = QVBoxLayout()
        self.easting_text = QLineEdit()
        self.easting_text.setToolTip(
            "Python list of column names to try as the Easting (X coordinate) column for\n"
            "georeferencing. If a column named 'WKT' is present it takes priority over this.\n\n"
            "Example: ['easting']"
        )
        easting_layout.addWidget(self.easting_text)
        easting_group.setLayout(easting_layout)
        layout.addWidget(easting_group)

        # Northing Column preferences
        northing_group = QGroupBox("Northing Column Preferences. (WKT takes preference)")
        northing_layout = QVBoxLayout()
        self.northing_text = QLineEdit()
        self.northing_text.setToolTip(
            "Python list of column names to try as the Northing (Y coordinate) column for\n"
            "georeferencing. If a column named 'WKT' is present it takes priority over this.\n\n"
            "Example: ['northing']"
        )
        northing_layout.addWidget(self.northing_text)
        northing_group.setLayout(northing_layout)
        layout.addWidget(northing_group)

        # Version regex
        version_group = QGroupBox("Version Regex")
        version_layout = QVBoxLayout()
        self.version_regex_edit = QLineEdit()
        self.version_regex_edit.setToolTip(
            "Regular expression used to detect and strip the version suffix from filenames.\n"
            "Must contain exactly one capturing group ( ) that marks the version token.\n"
            "The entire match is removed when stripping; the new version string from the\n"
            "Version field is then appended with a leading underscore.\n\n"
            "Examples:\n"
            "  _(v[\\d]+)   matches _v02, _v003, etc.  (default)\n"
            "  _([\\d]{3})  matches _001, _002, etc.\n"
            "  _(rev\\d+)   matches _rev1, _rev12, etc."
        )
        version_layout.addWidget(self.version_regex_edit)
        version_group.setLayout(version_layout)
        layout.addWidget(version_group)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate_and_accept(self):
        """Validate all fields before accepting the dialog"""
        pattern = self.version_regex_edit.text().strip()
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            QMessageBox.critical(self, "Invalid Regex", f"Version regex is not a valid regular expression:\n{e}")
            return
        if compiled.groups != 1:
            QMessageBox.critical(
                self, "Invalid Regex",
                f"Version regex must contain exactly one capturing group ( ).\n"
                f"Found {compiled.groups} group(s) in: {pattern}"
            )
            return
        self.accept()

    def set_values(self, x_prefs, y_prefs, n_prefs, x_unsortable_prefs, easting_prefs, northing_prefs, version_regex):
        """Set the dialog values"""
        self.x_text.setText(str(x_prefs))
        self.y_text.setText(str(y_prefs))
        self.n_text.setText(str(n_prefs))
        self.x_unsortable_text.setText(str(x_unsortable_prefs))
        self.easting_text.setText(str(easting_prefs))
        self.northing_text.setText(str(northing_prefs))
        self.version_regex_edit.setText(version_regex)

    def get_values(self):
        """Get the dialog values. Returns None values on parse error."""
        try:
            x_prefs = literal_eval(self.x_text.text())
            y_prefs = literal_eval(self.y_text.text())
            n_prefs = literal_eval(self.n_text.text())
            x_unsortable_prefs = literal_eval(self.x_unsortable_text.text())
            easting_prefs = literal_eval(self.easting_text.text())
            northing_prefs = literal_eval(self.northing_text.text())
            version_regex = self.version_regex_edit.text().strip()
            return x_prefs, y_prefs, n_prefs, x_unsortable_prefs, easting_prefs, northing_prefs, version_regex
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error parsing preferences: {str(e)}")
            return None, None, None, None, None, None, None

class CrossSectionEditorApp(QMainWindow):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        Processing.initialize()

        # Store the iface reference
        self.iface = iface

        self.setWindowTitle("Cross Section Editor")
        self.setGeometry(100, 100, 1280, 720)

        # State variables
        self.current_file_index = -1
        self.file_path = None
        self.file_name = None
        self.file_list = []
        self.csv_files = []
        self.left_bank = None
        self.right_bank = None
        self.left_bank_index = None
        self.right_bank_index = None
        self.current_data = None
        self.x_column = None
        self.y_column = None
        self.n_column = None
        self.has_header = True
        self.interpolated_left_idx = None
        self.interpolated_right_idx = None

        # Column preferences (default)
        self.x_column_preferences = ['x', 'x (m)', 'chainage', 'w', 0]
        self.y_column_preferences = ['y', 'z', 'h', 1]
        self.n_column_preferences = ['n', 'm', 'Mannings n']
        self.x_column_unsortable_preferences = ['w']
        self.easting_column_preferences = ['easting']
        self.northing_column_preferences = ['northing']

        # Cross section overlaps with SHP/GPKG
        self.polygon_layer = None
        self.paths_layer = None
        self.overlaps = []
        self.paths_style = os.path.join(os.path.dirname(__file__), 'styles', 'paths_style.qml')
        self.num_paths = None

        # Plot
        if matplotlib.__version__ >= '3.7.0':
            self.plot_loc_to_use = 'outside lower center'
        else:
            self.plot_loc_to_use = 'lower center'
        self.marker_points = None
        self._hover_cid = None

        # Version regex used to strip/detect version suffixes in filenames
        self.version_regex = r'_(v[\d]+)'

        # Other CSVs (version comparison overlay)
        self.all_csv_files: list = []       # master list of every loaded path
        self.version_link_map: dict = {}    # active_path -> comparison_path (rebuilt by recompute_version_links)
        self.other_version_csv = None
        self.other_version_csv_name = None
        self.other_version_csv_x = None
        self.other_version_csv_y = None

        # Create the UI
        self.init_ui()

    def init_ui(self):
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Main layout
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Header with controls (fixed height)
        header_widget = QWidget()
        header_widget.setMaximumHeight(150)
        header_layout = QGridLayout(header_widget)
        header_layout.setContentsMargins(5, 5, 5, 5)
        header_layout.setSpacing(5)

        # Previous and Next buttons
        self.prev_button = QPushButton("Previous")
        self.prev_button.setToolTip("Load the previous CSV file in the list")
        self.prev_button.clicked.connect(self.previous_file)
        header_layout.addWidget(self.prev_button, 0, 0)

        self.next_button = QPushButton("Next")
        self.next_button.setToolTip("Load the next CSV file in the list")
        self.next_button.clicked.connect(self.next_file)
        header_layout.addWidget(self.next_button, 0, 1)

        # File name label
        self.filename_label = QLabel("No file loaded")
        header_layout.addWidget(self.filename_label, 0, 2, 1, 2)

        self.reload_button = QPushButton("Reload File")
        self.reload_button.setToolTip("Discard unsaved changes and reload the current file from disk")
        self.reload_button.clicked.connect(self.reload_current_file)
        header_layout.addWidget(self.reload_button, 0, 4)

        # Save button
        self.save_button = QPushButton("Save")
        self.save_button.setToolTip("Save the current file (applies bank trimming and StartX=0 offset if enabled)")
        self.save_button.clicked.connect(self.save_file)
        header_layout.addWidget(self.save_button, 0, 5)

        # Undo/Redo buttons
        self.undo_button = QPushButton("Undo")
        self.undo_button.setToolTip("Undo the last edit (not yet implemented)")
        self.undo_button.clicked.connect(self.undo)
        self.undo_button.setEnabled(False)  # Disabled until implemented
        header_layout.addWidget(self.undo_button, 1, 0)

        self.redo_button = QPushButton("Redo")
        self.redo_button.setToolTip("Redo the last undone edit (not yet implemented)")
        self.redo_button.clicked.connect(self.redo)
        self.redo_button.setEnabled(False)  # Disabled until implemented
        header_layout.addWidget(self.redo_button, 1, 1)

        # Version control options
        self.version_combo = QComboBox()
        self.version_combo.setToolTip(
            "Increment Version: saves to a new file with the version string appended/replaced\n"
            "Change in-place: overwrites the existing file"
        )
        self.version_combo.addItems(["Increment Version", "Change in-place"])
        header_layout.addWidget(self.version_combo, 1, 2)

        self.version_label = QLabel("Version:")
        header_layout.addWidget(self.version_label, 1, 3)
        header_layout.setAlignment(self.version_label, Qt.AlignmentFlag.AlignRight)

        self.version_edit = QLineEdit("v02")
        self.version_edit.setFixedWidth(50)
        self.version_edit.setToolTip("Version string to append when saving with 'Increment Version' (e.g. v02, v003)")
        header_layout.addWidget(self.version_edit, 1, 4)
        header_layout.setAlignment(self.version_edit, Qt.AlignmentFlag.AlignLeft)

        # Initially hide the label and line edit
        self.version_label.setVisible(self.version_combo.currentText() == "Increment Version")
        self.version_edit.setVisible(self.version_combo.currentText() == "Increment Version")

        # Connect signal to slot
        self.version_combo.currentIndexChanged.connect(self.toggle_version_fields)

        # Column Settings button
        self.column_settings_button = QPushButton("Settings")
        self.column_settings_button.setToolTip("Configure column name preferences, unsortable columns, and version regex")
        self.column_settings_button.clicked.connect(self.show_settings)
        header_layout.addWidget(self.column_settings_button, 1, 5)

        # Options
        # ---

        # Fix verticals
        self.fix_verticals_check = QCheckBox("Fix Verticals and order on load")
        self.fix_verticals_check.setToolTip(
            "On load: sort rows by X value and offset any duplicate X values by a small amount\n"
            "so the cross section has strictly increasing chainage."
        )
        header_layout.addWidget(self.fix_verticals_check, 2, 0)
        self.fix_verticals_check.stateChanged.connect(self.reload_current_file)

        # Make left most active point X=0
        self.make_leftmost_zero_check = QCheckBox("Open and save with StartX=0")
        self.make_leftmost_zero_check.setToolTip(
            "Shift the active (in-bank) cross section so it starts at X=0.\n"
            "Trimmed rows outside the banks will have negative X values.\n"
            "The same offset is applied on save so the file on disk also starts at zero."
        )
        header_layout.addWidget(self.make_leftmost_zero_check, 2, 1)
        self.make_leftmost_zero_check.stateChanged.connect(self.reload_current_file)

        self.autosave_check = QCheckBox("Autosave on section change")
        self.autosave_check.setToolTip("Automatically save the current file before switching to another file in the list")
        header_layout.addWidget(self.autosave_check, 2, 2)

        self.make_plot_file_check = QCheckBox("Make plot file on save")
        self.make_plot_file_check.setToolTip("Generate a .png plot image alongside the CSV whenever the file is saved")
        header_layout.addWidget(self.make_plot_file_check, 2, 3)

        self.polygon_layer_combo = QComboBox()
        self.polygon_layer_combo.setToolTip("Select a polygon layer loaded in QGIS")
        header_layout.addWidget(self.polygon_layer_combo, 2, 4)
        self.polygon_layer_combo.currentIndexChanged.connect(self.on_polygon_layer_changed)

        # Keep combo in sync with project layer changes
        project = QgsProject.instance()
        assert project is not None
        project.layersAdded.connect(self.refresh_polygon_layer_combo)
        project.layersRemoved.connect(self.refresh_polygon_layer_combo)

        self.refresh_polygon_layer_combo()

        self.version_link_combo = QComboBox()
        self.version_link_combo.addItems([
            "No version linking",
            "Prefer newer (consecutive)",
            "Prefer older (consecutive)",
            "Prefer newest (extreme)",
            "Prefer oldest (extreme)",
        ])
        self.version_link_combo.setToolTip(
            "Controls how versioned CSV files already in the list are paired for comparison.\n\n"
            "No version linking: each file is shown and navigated independently.\n\n"
            "Prefer newer (consecutive): each file paired with its immediate predecessor.\n"
            "  Newer file is active; older is the red overlay.\n"
            "  e.g. v01→v02, v02→v10  [v01 hidden, v02 compares to v01, v10 compares to v02]\n\n"
            "Prefer older (consecutive): each file paired with its immediate successor.\n"
            "  Older file is active; newer is the overlay.\n"
            "  e.g. v01→v02, v02→v10  [v10 hidden, v01 compares to v02, v02 compares to v10]\n\n"
            "Prefer newest (extreme): one entry per group — newest is active, oldest is overlay.\n"
            "  e.g. Test_v10.csv (Test_v01)  [middle versions hidden]\n\n"
            "Prefer oldest (extreme): one entry per group — oldest is active, newest is overlay.\n"
            "  e.g. Test_v01.csv (Test_v10)  [middle versions hidden]\n\n"
            "Version pairing uses the Version Regex from Settings to strip the suffix.\n"
            "Closing a linked entry also removes its hidden comparison file from the list."
        )
        header_layout.addWidget(self.version_link_combo, 2, 5)
        self.version_link_combo.currentIndexChanged.connect(self.recompute_version_links)

        # Static helper text
        self.left_bank_ctrl = QLabel("Ctrl + Click on plot to set Left Bank (Snapped)")
        header_layout.addWidget(self.left_bank_ctrl, 3, 0, 1, 2)
        self.right_bank_alt = QLabel("Alt + Click on plot to set Right Bank (Snapped)")
        header_layout.addWidget(self.right_bank_alt, 3, 2, 1, 2)
        self.table_right_click = QLabel("Right Click on table/plot view to set banks (Interpolated)")
        header_layout.addWidget(self.table_right_click, 3, 4, 1, 2)

        # Add the header to the main layout
        main_layout.addWidget(header_widget)

        # Splitter for the three main sections
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Table view (left)
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self.table_view = QTableView()
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_view.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_view.horizontalHeader().customContextMenuRequested.connect(self.show_header_context_menu)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        # Enable context menu for table view
        self.table_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self.show_context_menu)

        table_layout.addWidget(self.table_view)

        # Plot (middle)
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = MatplotlibCanvas(plot_widget, width=5, height=4, dpi=100)
        plot_layout.addWidget(self.canvas)

        # Add toolbar
        self.toolbar = NavigationToolbar(self.canvas, plot_widget)
        plot_layout.addWidget(self.toolbar)

        # File list (right)
        file_list_widget = QWidget()
        file_list_layout = QVBoxLayout(file_list_widget)
        file_list_layout.setContentsMargins(0, 0, 0, 0)

        file_list_label = QLabel("CSV Files")
        file_list_layout.addWidget(file_list_label)

        self.file_list_widget = QListWidget()
        self.file_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_list_widget.itemClicked.connect(self.on_file_selected)
        file_list_layout.addWidget(self.file_list_widget)

        # Load files button
        self.load_files_button = QPushButton("Load CSV Files")
        self.load_files_button.setToolTip("Open one or more CSV files and add them to the list")
        self.load_files_button.clicked.connect(self.load_csv_files)
        file_list_layout.addWidget(self.load_files_button)

        # Close selected files button
        self.close_all_files_button = QPushButton("Close Selected")
        self.close_all_files_button.setToolTip("Remove the selected file(s) from the list (Ctrl+A to select all)")
        self.close_all_files_button.clicked.connect(self.close_selected_csv_files)
        file_list_layout.addWidget(self.close_all_files_button)

        # Add the three widgets to the splitter
        splitter.addWidget(table_widget)
        splitter.addWidget(plot_widget)
        splitter.addWidget(file_list_widget)

        # Set the initial sizes of the splitter
        splitter.setSizes([int(self.width()/3), int(self.width()/3), int(self.width()/3)])

        # Add the splitter to the main layout
        main_layout.addWidget(splitter)

        # Status bar for messages
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.message_queue = deque(maxlen=5)
        self.current_message_timer = QTimer(self)
        self.current_message_timer.timeout.connect(self.show_next_message)

        # Set up event handling for the plot
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)

        # Update marker points
        self.marker_points = None

        # Show the UI
        self.show()

    def closeEvent(self, event):
        """Remove the temporary paths layer from the QGIS project on close."""
        if self.paths_layer and self.paths_layer.isValid():
            project = QgsProject.instance()
            if project and self.paths_layer.id() in project.mapLayers():
                project.removeMapLayer(self.paths_layer)
        self.paths_layer = None
        super().closeEvent(event)

    def show_status_message(self, message, duration=3000):
        """Queue a message and show it in the status bar."""
        self.message_queue.append((message, duration))

        # If no message is currently being displayed, start showing messages
        if not self.current_message_timer.isActive():
            self.show_next_message()

    def show_next_message(self):
        """Display the next message in the queue."""
        if self.message_queue:
            message, duration = self.message_queue.popleft()
            self.status_bar.showMessage(message, duration)
            self.current_message_timer.start(duration)  # Wait before showing next
        else:
            self.current_message_timer.stop()  # No more messages, stop timer

    def toggle_version_fields(self):
        """Enable/disable the version label and edit box without shifting the layout."""
        is_increment = self.version_combo.currentText() == "Increment Version"

        # Instead of hiding, disable them and make them transparent
        self.version_label.setEnabled(is_increment)
        self.version_edit.setEnabled(is_increment)

        # Adjust background and text color when hidden
        if is_increment:
            self.version_edit.setStyleSheet("")  # Reset to default
        else:
            self.version_edit.setStyleSheet("background: transparent; border: none; color: transparent;")

    def show_context_menu(self, position: QPoint):
        """Show context menu when right-clicking on a table row"""
        # print("Context menu requested at position:", position)
        index = self.table_view.indexAt(position)
        # print("Index valid:", index.isValid())
        # print("Current data is None:", self.current_data is None)
        # print("X column is None:", self.x_column is None)
        if not index.isValid() or self.current_data is None or self.x_column is None:
            # print("Early return because of failed conditions")
            return  # Do nothing if right-click is outside the table

        row = index.row()
        x_value = self.current_data.iloc[row][self.x_column]  # Get X value from row
        # print(f"x_value: {x_value}")

        # Create context menu
        menu = QMenu(self)
        # print(f"menu: {menu}")

        # Add actions
        left_action = QAction("Set Left Bank", self)
        right_action = QAction("Set Right Bank", self)
        # print(f"left_action: {left_action}")

        # Connect actions directly to the appropriate methods
        left_action.triggered.connect(lambda: self.set_left_bank(x_value, row))
        right_action.triggered.connect(lambda: self.set_right_bank(x_value, row))

        # Add actions to menu
        menu.addAction(left_action)
        menu.addAction(right_action)
        # print(f"menu: {menu}")

        # Show menu at cursor position
        # print(f"position = self.table_view.viewport().mapToGlobal(position): {self.table_view.viewport().mapToGlobal(position)}")
        menu.exec(self.table_view.viewport().mapToGlobal(position))

    def show_settings(self):
        """Show the settings dialog"""
        dialog = SettingsDialog(self)
        dialog.set_values(
            self.x_column_preferences, self.y_column_preferences,
            self.n_column_preferences, self.x_column_unsortable_preferences,
            self.easting_column_preferences, self.northing_column_preferences,
            self.version_regex
        )

        if dialog.exec():
            x_prefs, y_prefs, n_prefs, x_unsortable_prefs, easting_prefs, northing_prefs, version_regex = dialog.get_values()
            if x_prefs is not None:
                self.x_column_preferences = x_prefs
                self.y_column_preferences = y_prefs
                self.n_column_preferences = n_prefs
                self.x_column_unsortable_preferences = x_unsortable_prefs
                self.easting_column_preferences = easting_prefs
                self.northing_column_preferences = northing_prefs
                self.version_regex = version_regex

                # Reload the current file with new column settings
                if self.current_data is not None:
                    self.reload_current_file()

    def show_header_context_menu(self, pos):
        """Show context menu for header clicks"""
        if self.current_data is None:
            return

        # Get the column index
        index = self.table_view.horizontalHeader().logicalIndexAt(pos)
        if index >= 0:
            # Create context menu
            context_menu = QMenu(self)

            # Add actions
            set_x_action = QAction(f"Set as X Column", self)
            set_x_action.triggered.connect(lambda: self.set_column_as_x(index))
            context_menu.addAction(set_x_action)

            set_y_action = QAction(f"Set as Y Column", self)
            set_y_action.triggered.connect(lambda: self.set_column_as_y(index))
            context_menu.addAction(set_y_action)

            set_n_action = QAction(f"Set as N Column", self)
            set_n_action.triggered.connect(lambda: self.set_column_as_n(index))
            context_menu.addAction(set_n_action)

            # Show the menu
            context_menu.exec(self.table_view.horizontalHeader().mapToGlobal(pos))

    def set_column_as_x(self, index):
        """Set the specified column as X column"""
        if self.current_data is not None:
            self.x_column = self.current_data.columns[index]
            self.update_plot()
            self.show_status_message(f"Set {self.x_column} as X column", 1000)

    def set_column_as_y(self, index):
        """Set the specified column as Y column"""
        if self.current_data is not None:
            self.y_column = self.current_data.columns[index]
            self.update_plot()
            self.show_status_message(f"Set {self.y_column} as Y column", 1000)

    def set_column_as_n(self, index):
        """Set the specified column as N column"""
        if self.current_data is not None:
            self.n_column = self.current_data.columns[index]
            self.update_plot()
            self.show_status_message(f"Set {self.n_column} as N column", 1000)

    def load_csv_files(self):
        """Open file dialog to select CSV files"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("CSV files (*.csv)")

        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()

            # Ensure uniqueness while preserving order; exclude sidecar trim files
            existing = set(self.all_csv_files)
            unique_new = [f for f in file_paths if f not in existing and not f.endswith('.trim.csv')]

            if unique_new:
                self.all_csv_files.extend(unique_new)
                self.recompute_version_links()

    def close_selected_csv_files(self):
        """Close the files selected in the CSV list and update the UI"""
        selected_rows = sorted(
            [self.file_list_widget.row(item) for item in self.file_list_widget.selectedItems()],
            reverse=True
        )
        if not selected_rows:
            return

        current_path = (
            self.csv_files[self.current_file_index]
            if 0 <= self.current_file_index < len(self.csv_files)
            else None
        )
        current_removed = self.current_file_index in selected_rows

        # Collect paths to remove from the master list (active file + its hidden comparison)
        paths_to_remove: set = set()
        for row in selected_rows:
            active_path = self.csv_files[row]
            paths_to_remove.add(active_path)
            comparison = self.version_link_map.get(active_path)
            if comparison:
                paths_to_remove.add(comparison)

        self.all_csv_files = [f for f in self.all_csv_files if f not in paths_to_remove]

        n = len(selected_rows)

        # recompute_version_links rebuilds csv_files and file_list_widget
        self.recompute_version_links()

        if current_removed or not self.csv_files:
            self.current_file_index = -1
            self.current_data = None
            self.left_bank = self.right_bank = None
            self.left_bank_index = self.right_bank_index = None
            self.other_version_csv = None
            self.other_version_csv_name = None
            self.other_version_csv_x = None
            self.other_version_csv_y = None
            self.table_view.setModel(None)
            self.canvas.axes.cla()
            self.canvas.draw_idle()
            self.filename_label.setText("No file loaded")
        elif current_path in self.csv_files:
            self.current_file_index = self.csv_files.index(current_path)
            self.file_list_widget.setCurrentRow(self.current_file_index)

        self.show_status_message(f"Closed {n} file{'s' if n != 1 else ''}", 1000)

    def recompute_version_links(self):
        """Rebuild csv_files, version_link_map, file_list, and file_list_widget from all_csv_files.

        Linking mode (from version_link_combo):
          0 – No version linking         : csv_files == all_csv_files; no overlay
          1 – Prefer newer (consecutive) : active = newer, overlay = immediate predecessor
          2 – Prefer older (consecutive) : active = older, overlay = immediate successor
          3 – Prefer newest (extreme)    : active = newest, overlay = oldest (one entry per group)
          4 – Prefer oldest (extreme)    : active = oldest, overlay = newest (one entry per group)
        """
        mode = self.version_link_combo.currentIndex()
        regex = self.version_regex or r'_(v[\d]+)'

        def strip_version(path: str) -> str:
            base = os.path.splitext(os.path.basename(path))[0]
            return re.sub(regex, '', base)

        def version_key(path: str):
            """Return a sortable key from the version-token digits, or () for no match."""
            base = os.path.splitext(os.path.basename(path))[0]
            m = re.search(regex, base)
            if m:
                digits = re.sub(r'[^\d]', '', m.group(1))
                return tuple(int(c) for c in digits) if digits else ()
            return ()

        self.version_link_map = {}

        # Always work in alphabetical-basename order so newly added files slot in correctly
        sorted_all = sorted(self.all_csv_files, key=lambda p: os.path.basename(p).lower())

        if mode == 0:
            # No linking — every file is independent and visible
            self.csv_files = sorted_all
        else:
            # Group files by their base name (version suffix stripped)
            groups: dict[str, list] = {}
            for path in sorted_all:
                key = strip_version(path)
                groups.setdefault(key, []).append(path)

            active_files: list = []

            for group in groups.values():
                if len(group) < 2:
                    # Single file in group — no linking possible
                    active_files.append(group[0])
                    continue

                # Sort ascending by version
                sorted_group = sorted(group, key=version_key)

                if mode == 1:
                    # Prefer newer (consecutive): active = newer, overlay = immediate predecessor
                    # e.g. [v01, v02, v10] → v02(active,v01), v10(active,v02); v01 hidden
                    for i, path in enumerate(sorted_group):
                        if i == 0:
                            pass  # oldest becomes overlay only, not active
                        else:
                            active_files.append(path)
                            self.version_link_map[path] = sorted_group[i - 1]

                elif mode == 2:
                    # Prefer older (consecutive): active = older, overlay = immediate successor
                    # e.g. [v01, v02, v10] → v01(active,v02), v02(active,v10); v10 hidden
                    for i, path in enumerate(sorted_group):
                        if i == len(sorted_group) - 1:
                            pass  # newest becomes overlay only, not active
                        else:
                            active_files.append(path)
                            self.version_link_map[path] = sorted_group[i + 1]

                elif mode == 3:
                    # Prefer newer (extreme): one entry — newest active, oldest overlay
                    # e.g. [v01, v02, v10] → v10(active,v01); v01,v02 hidden
                    newest = sorted_group[-1]
                    oldest = sorted_group[0]
                    active_files.append(newest)
                    self.version_link_map[newest] = oldest

                else:
                    # mode == 4 — Prefer older (extreme): one entry — oldest active, newest overlay
                    # e.g. [v01, v02, v10] → v01(active,v10); v02,v10 hidden
                    oldest = sorted_group[0]
                    newest = sorted_group[-1]
                    active_files.append(oldest)
                    self.version_link_map[oldest] = newest

            # Preserve alphabetical order for active files
            order = {p: i for i, p in enumerate(sorted_all)}
            self.csv_files = sorted(active_files, key=lambda p: order.get(p, 0))

        # Rebuild display names (show full path only when basenames collide)
        name_counts: dict[str, int] = {}
        for f in self.csv_files:
            name_counts[os.path.basename(f)] = name_counts.get(os.path.basename(f), 0) + 1

        def display_name(path: str) -> str:
            name = path if name_counts[os.path.basename(path)] > 1 else os.path.basename(path)
            comparison = self.version_link_map.get(path)
            if comparison:
                name += f" ({os.path.splitext(os.path.basename(comparison))[0]})"
            return name

        self.file_list = [display_name(f) for f in self.csv_files]

        self.file_list_widget.clear()
        self.file_list_widget.addItems(self.file_list)

        # Restore the highlighted selection after the widget was repopulated
        current_path = (
            self.csv_files[self.current_file_index]
            if 0 <= self.current_file_index < len(self.csv_files)
            else None
        )
        if current_path and current_path in self.csv_files:
            self.current_file_index = self.csv_files.index(current_path)
            self.file_list_widget.setCurrentRow(self.current_file_index)

    def load_other_csv_file(self):
        """Load the other csv file into df for plot"""
        self.other_version_csv_x = None
        self.other_version_csv_y = None

        assert self.other_version_csv is not None
        try:
            has_header = self.detect_header(self.other_version_csv)
            other_df = pd.read_csv(self.other_version_csv, index_col=None, header=0 if has_header else None, comment='!')

            # Ensure numeric column indices are treated as integers
            try:
                other_df.columns = [int(col) if str(col).isdigit() else col for col in other_df.columns]
            except ValueError:
                pass  # Some columns are non-numeric, ignore conversion failure

            # Detect X and Y columns *after* reading the full data
            x_column, y_column, _ = self.detect_xy_columns(other_df)

            self.other_version_csv_x = other_df[x_column]
            self.other_version_csv_y = other_df[y_column]

            self.show_status_message(f"Loaded other: {self.other_version_csv_name}", 500)

        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"Other file not found: {self.other_version_csv}")
        except pd.errors.EmptyDataError:
            QMessageBox.critical(self, "Error", f"The other file is empty: {self.other_version_csv}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading other file: {str(e)}")

    def load_current_file(self):
        """Load the current file into the table and plot"""
        if 0 <= self.current_file_index < len(self.csv_files):
            try:
                self.interpolated_left_idx = None
                self.interpolated_right_idx = None

                # Load the CSV file
                self.file_path = self.csv_files[self.current_file_index]
                self.file_name = os.path.splitext(os.path.basename(self.file_path))[0]

                # Resolve comparison overlay from version link map
                comparison_path = self.version_link_map.get(self.file_path)
                if comparison_path:
                    self.other_version_csv = comparison_path
                    self.other_version_csv_name = os.path.splitext(os.path.basename(comparison_path))[0]
                    self.load_other_csv_file()
                else:
                    self.other_version_csv = None
                    self.other_version_csv_name = None
                    self.other_version_csv_x = None
                    self.other_version_csv_y = None

                # Check if the first row contains column-like names
                self.has_header = self.detect_header(self.file_path)

                # Read the full CSV with correct header setting
                df = pd.read_csv(self.file_path, index_col=None, header=0 if self.has_header else None)

                # Ensure numeric column indices are treated as integers
                try:
                    df.columns = [int(col) if str(col).isdigit() else col for col in df.columns]
                except ValueError:
                    pass  # Some columns are non-numeric, ignore conversion failure

                # Detect X and Y columns *after* reading the full data
                self.x_column, self.y_column, self.n_column = self.detect_xy_columns(df)
                
                # Strip any legacy '!# ' markers left by old saves (first column only)
                first_col = df.columns[0]
                if not pd.api.types.is_numeric_dtype(df[first_col].dtype):
                    _legacy_pat = re.compile(r'^\s*[!#]{1,2}\s*')
                    df[first_col] = df[first_col].apply(
                        lambda v: _legacy_pat.sub('', str(v)) if pd.notna(v) else v
                    )
                # Drop legacy Trim column if present
                if 'Trim' in df.columns:
                    df = df.drop(columns=['Trim'])

                # Check for a .trim.csv sidecar produced by the current save format
                trim_path = os.path.splitext(self.file_path)[0] + '.trim.csv'
                if os.path.exists(trim_path):
                    trim_df = pd.read_csv(trim_path, index_col=None)
                    # Apply same integer-column normalisation as the main file
                    try:
                        trim_df.columns = [int(col) if str(col).isdigit() else col for col in trim_df.columns]
                    except ValueError:
                        pass

                    # Capture in-bank boundaries before combining
                    main_x      = pd.to_numeric(df[self.x_column], errors='coerce')
                    trim_x      = pd.to_numeric(trim_df[self.x_column], errors='coerce')
                    main_x_min  = float(main_x.min())
                    main_x_max  = float(main_x.max())
                    has_left_trim  = bool((trim_x < main_x_min).any())
                    has_right_trim = bool((trim_x > main_x_max).any())

                    # Combine and sort so the full cross-section is available for display
                    df = pd.concat([trim_df, df], ignore_index=True)
                    df = df.sort_values(by=self.x_column).reset_index(drop=True)

                    # make_leftmost_zero: shift by the in-bank minimum so the active
                    # section starts at zero; trim rows on the left become negative.
                    if self.make_leftmost_zero_check.isChecked():
                        df[self.x_column] = pd.to_numeric(df[self.x_column], errors='coerce') - main_x_min
                        main_x_max = main_x_max - main_x_min
                        main_x_min = 0.0

                    # Set bank markers
                    if has_left_trim:
                        lb_idx = df.index[pd.to_numeric(df[self.x_column], errors='coerce') == main_x_min]
                        self.left_bank       = main_x_min
                        self.left_bank_index = int(lb_idx[0]) if len(lb_idx) else None
                    else:
                        self.left_bank       = None
                        self.left_bank_index = None

                    if has_right_trim:
                        rb_idx = df.index[pd.to_numeric(df[self.x_column], errors='coerce') == main_x_max]
                        self.right_bank       = main_x_max
                        self.right_bank_index = int(rb_idx[-1]) if len(rb_idx) else None
                    else:
                        self.right_bank       = None
                        self.right_bank_index = None

                else:
                    # No sidecar — no banks
                    self.left_bank = self.right_bank = None
                    self.left_bank_index = self.right_bank_index = None

                    # Apply fix verticals if needed
                    if self.fix_verticals_check.isChecked():
                        df = self.fix_verticals(df)

                    # make_leftmost_zero shifts by global min (no trim data present)
                    if self.make_leftmost_zero_check.isChecked():
                        df = self.make_leftmost_zero(df)

                # Set the current data
                self.current_data = df.copy()

                # Select the current file in the list
                self.file_list_widget.setCurrentRow(self.current_file_index)

                # Try to get instersections with polygon SHP/GPKG
                if self.polygon_layer:
                    # print(f"self.polygon_layer: {self.polygon_layer}")
                    if self.polygon_layer.isValid():
                        self.points_to_path()
                    else:
                        self.show_status_message(f"self.polygon_layer.isValid(): {self.polygon_layer.isValid()}")

                if self.paths_layer:
                    # print(f"self.paths_layer: {self.paths_layer}")
                    if self.paths_layer.isValid():
                        self.find_overlap_with_polygon()
                    else:
                        self.show_status_message(f"self.paths_layer.isValid(): {self.paths_layer.isValid()}")

                # Update the UI
                self.update_table()
                self.update_plot(clear=True)

                self.filename_label.setText(os.path.basename(self.file_path))
                self.show_status_message(f"Loaded: {os.path.basename(self.file_path)}", 500)

            except FileNotFoundError:
                QMessageBox.critical(self, "Error", f"File not found: {self.file_path}")
            except pd.errors.EmptyDataError:
                QMessageBox.critical(self, "Error", f"The file is empty: {self.file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")

    def reload_current_file(self):
        """Reload the current file with new settings"""
        if 0 <= self.current_file_index < len(self.csv_files):
            self.load_current_file()

    def detect_header(self, file_path):
        """Determine if the CSV file has a header by analyzing the first few non-comment rows."""
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read lines and filter out comments starting with '!'
            valid_lines = [line for line in f if not line.strip().startswith('!')]

        # If not enough data to check, assume no header
        if len(valid_lines) < 2:
            return False

        # Use StringIO to treat filtered lines as a file-like object for pandas
        sample_data = StringIO(''.join(valid_lines[:3]))  # check first 3 non-comment lines
        sample_df = pd.read_csv(sample_data, header=None, dtype=str)

        first_row = sample_df.iloc[0]
        second_row = sample_df.iloc[1] if len(sample_df) > 1 else None

        # Check if first row is mostly strings while second row is mostly numeric
        first_row_strings = sum(first_row.apply(lambda x: isinstance(x, str) and not x.replace('.', '', 1).isdigit()))
        second_row_strings = sum(second_row.apply(lambda x: isinstance(x, str) and not x.replace('.', '', 1).isdigit())) if second_row is not None else 0

        has_header = first_row_strings > second_row_strings
        return has_header

    def detect_xy_columns(self, df):
        """Attempt to detect X and Y columns from preferences"""
        def match_column(preferences, df_columns):
            col_map = {str(c).lower(): c for c in df_columns}
            for pref in preferences:
                if isinstance(pref, int):
                    # This is a numeric index
                    if pref < len(df_columns):
                        return df_columns[pref]
                else:
                    # This is a column name
                    key = str(pref).lower()
                    if key in col_map:
                        return col_map[key]
            return None

        # Find X column
        x_column = match_column(self.x_column_preferences, df.columns)
        if x_column is None and not df.empty:
            QMessageBox.warning(self, "Warning", f"No X column from preferences matches datafile: {self.x_column_preferences}")

        # Find Y column
        y_column = match_column(self.y_column_preferences, df.columns)
        if y_column is None and not df.empty:
            QMessageBox.warning(self, "Warning", f"No Y column from preferences matches datafile: {self.y_column_preferences}")

        # Find N column
        n_column = match_column(self.n_column_preferences, df.columns)
        if n_column is None and not df.empty:
            self.show_status_message(f"No N column from preferences matches datafile: {self.n_column_preferences}", 2000)

        return x_column, y_column, n_column

    def update_table(self):
        """Update the table view with current data"""
        if self.current_data is not None:
            # print(f"self.current_data: {self.current_data}")
            # Create model
            model = EditablePandasModel(self.current_data)

            # Set model first (this will trigger signals)
            self.table_view.setModel(model)

            # Connect data change signal AFTER setting the model
            model.dataChanged.connect(self.on_table_data_changed)

            # Update cut indices
            self.update_cut_indices(model)

    def update_cut_indices(self, model=None):
        """Update indices that will be cut"""
        if DEBUG: print("[DEBUG update_cut_indices] entered", flush=True)
        if model is None:
            model = self.table_view.model()
            if model is None:
                if DEBUG: print("[DEBUG update_cut_indices] no model, returning", flush=True)
                return

        if self.current_data is not None and self.x_column:
            cut_indices = []

            if self.left_bank is not None:
                left_cut = self.current_data[self.current_data[self.x_column] < self.left_bank].index.tolist()
                cut_indices.extend(left_cut)
                if DEBUG: print(f"[DEBUG update_cut_indices] left cut rows={len(left_cut)}", flush=True)

            if self.right_bank is not None:
                right_cut = self.current_data[self.current_data[self.x_column] > self.right_bank].index.tolist()
                cut_indices.extend(right_cut)
                if DEBUG: print(f"[DEBUG update_cut_indices] right cut rows={len(right_cut)}", flush=True)

            if DEBUG: print(f"[DEBUG update_cut_indices] calling model.set_cut_indices() with {len(cut_indices)} rows", flush=True)
            model.set_cut_indices(cut_indices)
            if DEBUG: print("[DEBUG update_cut_indices] done", flush=True)

    def on_table_data_changed(self):
        """Handle data changes in the table"""
        if DEBUG: print("[DEBUG on_table_data_changed] entered", flush=True)
        model = self.table_view.model()
        if model:
            self.current_data = model.get_dataframe()
            if DEBUG: print(f"[DEBUG on_table_data_changed] got dataframe shape={self.current_data.shape}, calling update_plot()", flush=True)
            self.update_plot(preserve_view=True)
            if DEBUG: print("[DEBUG on_table_data_changed] done", flush=True)


    def update_plot(self, preserve_view=False, clear=False):
        """Update the plot with current data

        Args:
            preserve_view (bool): If True, preserve the current axis limits
            clear (bool): If True, clear the existing plot
        """
        if clear is True:
            self.canvas.axes.cla()

        # Store current view limits if needed
        if preserve_view:
            xlim = self.canvas.axes.get_xlim()
            ylim = self.canvas.axes.get_ylim()

        if self.current_data is not None:
            # Consolidated data extraction and validation
            try:
                x = self.current_data[self.x_column]
                y = self.current_data[self.y_column]
            except KeyError as e:
                self.show_status_message(f"Column not found: {e}", 2000)
                return

            # Clear the plot
            self.canvas.axes.clear()

            # Create empty lists to store legend elements
            legend_elements = []
            legend_labels = []

            # Add section line to legend
            line, = self.canvas.axes.plot(x, y, '-', color='blue', alpha=0.5)
            legend_elements.append(line)
            legend_labels.append('Section')

            # Other CSV plot
            if self.other_version_csv is not None:
                other_x = self.other_version_csv_x
                other_y = self.other_version_csv_y
                line, = self.canvas.axes.plot(other_x, other_y, '-', color='red', alpha=0.5)
                legend_elements.append(line)
                legend_labels.append(f'Other: {self.other_version_csv_name}')

            # If n_column is provided and exists in the dataframe
            has_n_values = self.n_column and self.n_column in self.current_data.columns

            if has_n_values:
                n_values = self.current_data[self.n_column]
                # Handle categorical values
                unique_values = n_values.dropna().unique()
                colors = plt.get_cmap('tab10', len(unique_values))
                color_map = {val: colors(i) for i, val in enumerate(unique_values)}

                # Create separate collections for positive and negative n_values
                pos_mask = n_values >= 0
                neg_mask = n_values < 0

                # Plot positive values with circles
                if pos_mask.any():
                    pos_colors = [color_map[val] if val in color_map else 'gray' for val in n_values[pos_mask]]
                    scatter_pos = self.canvas.axes.scatter(
                        x[pos_mask], y[pos_mask],
                        c=pos_colors, marker='o', picker=5
                    )
                    legend_elements.append(scatter_pos)
                    legend_labels.append('+ve N/M')

                    # Enable hover annotations for positive points
                    self._setup_hover_annotations(scatter_pos, x[pos_mask], y[pos_mask],
                                                n_values[pos_mask] if has_n_values else None)

                # Plot negative values with X markers
                if neg_mask.any():
                    neg_colors =  [color_map[val] if val in color_map else 'gray' for val in n_values[neg_mask]]
                    scatter_neg = self.canvas.axes.scatter(
                        x[neg_mask], y[neg_mask],
                        c=neg_colors, marker='x', picker=5
                    )
                    legend_elements.append(scatter_neg)
                    legend_labels.append('-ve N/M')

                    # Enable hover annotations for negative points
                    self._setup_hover_annotations(scatter_neg, x[neg_mask], y[neg_mask],
                                                n_values[neg_mask] if has_n_values else None)

                # Store the scatter points (use the positive points collection if it exists, otherwise negative)
                if pos_mask.any():
                    self.marker_points = scatter_pos
                else:
                    self.marker_points = scatter_neg
            else:
                # Default: plot all points with circle markers
                scatter = self.canvas.axes.scatter(x, y, c='blue', marker='o', picker=5)
                self.marker_points = scatter
                # Enable hover annotations for all points (without n_values)
                self._setup_hover_annotations(scatter, x, y, None)

                # Add the default scatter to legend
                legend_elements.append(scatter)
                legend_labels.append('Points')

            # Bank indicators
            bank_marker = None

            # Shade left bank
            if self.left_bank is not None:
                min_y, max_y = min(y), max(y)
                min_x = min(x)
                self.canvas.axes.fill_betweenx(
                    [min_y, max_y], min_x, self.left_bank,
                    color='gray', alpha=0.3
                )
                if self.left_bank_index is not None:
                    left_x = x.iloc[self.left_bank_index]
                    left_y = y.iloc[self.left_bank_index]
                    bank_marker = self.canvas.axes.scatter(
                        left_x, left_y, c='black', marker='1', s=100, zorder=10
                    )

            # Shade right bank
            if self.right_bank is not None:
                min_y, max_y = min(y), max(y)
                max_x = max(x)
                self.canvas.axes.fill_betweenx(
                    [min_y, max_y], self.right_bank, max_x,
                    color='gray', alpha=0.3
                )
                if self.right_bank_index is not None:
                    right_x = x.iloc[self.right_bank_index]
                    right_y = y.iloc[self.right_bank_index]
                    if bank_marker is None:  # Only add once
                        bank_marker = self.canvas.axes.scatter(
                            right_x, right_y, c='black', marker='1', s=100, zorder=10
                        )
                    else:
                        self.canvas.axes.scatter(
                            right_x, right_y, c='black', marker='1', s=100, zorder=10
                        )

            # Add bank elements to legend if they exist
            if self.left_bank is not None or self.right_bank is not None:
                bank_patch = Patch(facecolor='gray', alpha=0.3)
                legend_elements.append(bank_patch)
                legend_labels.append('Banks')

                if bank_marker is not None:
                    legend_elements.append(bank_marker)
                    legend_labels.append('Banks Marker')

            # Shade SHP/GPKG
            polygon_patch = None
            if self.overlaps:
                min_y, max_y = min(y), max(y)
                min_x = min(x)
                for in_value, out_value in self.overlaps:
                    polygon_patch = self.canvas.axes.fill_betweenx(
                        [min_y, max_y], in_value + min_x, out_value + min_x,
                        color='lightblue', alpha=0.3
                    )

            # Add polygon patch to legend if it exists
            if polygon_patch is not None:
                polygon_legend = Patch(facecolor='lightblue', alpha=0.3)
                legend_elements.append(polygon_legend)
                legend_labels.append('Polygon')

            # Add labels
            self.canvas.axes.set_xlabel(self.x_column)
            self.canvas.axes.set_ylabel(self.y_column)
            self.canvas.axes.set_title(f"Cross Section: {self.file_list[self.current_file_index]}")
            self.canvas.axes.grid(True)

            # Add the legend below the axis with up to 3 columns
            if legend_elements:
                # Remove old legends
                for legend in self.canvas.figure.legends:
                    legend.remove()

                total_items = len(legend_elements)
                max_cols = 3
                num_cols = min(total_items, max_cols)

                self.canvas.figure.legend(
                    legend_elements,
                    legend_labels,
                    loc=self.plot_loc_to_use,
                    ncol=num_cols,
                    frameon=True,
                    fontsize='small'
                )

                plt.tight_layout()

            # Restore the previous view if requested
            if preserve_view:
                self.canvas.axes.set_xlim(xlim)
                self.canvas.axes.set_ylim(ylim)

            # Redraw
            self.canvas.draw_idle()

    def _disconnect_previous_hover_events(self):
        """Disconnect previous hover event handlers to prevent accumulation"""
        if hasattr(self, '_hover_cid') and self._hover_cid:
            try:
                self.canvas.mpl_disconnect(self._hover_cid)
                self._hover_cid = None
            except Exception as e:
                self.show_status_message(f"Error disconnecting hover event: {e}", 2000)

    def _setup_hover_annotations(self, scatter_points, x_values, y_values, n_values=None):
        """
        Setup hover annotations to display X, Y, and N values (if available) on hover

        Parameters:
        scatter_points: The matplotlib scatter collection
        x_values: X coordinate values corresponding to scatter points
        y_values: Y coordinate values corresponding to scatter points
        n_values: N values (optional) corresponding to scatter points
        """
        # Disconnect previous hover events first
        self._disconnect_previous_hover_events()

        # Create annotation object that will be updated
        annot = self.canvas.axes.annotate(
            "", xy=(0, 0), xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8, ec="black"),
            arrowprops=dict(arrowstyle="->")
        )
        annot.set_visible(False)

        def hover(event):
            # If the mouse is over the scatter points
            vis = annot.get_visible()
            if event.inaxes == self.canvas.axes:
                cont, ind = scatter_points.contains(event)
                if cont:
                    # Find the index of the point that was hovered over
                    ind_idx = ind["ind"][0]
                    # Update the position of the annotation
                    pos = scatter_points.get_offsets()[ind_idx]
                    annot.xy = pos

                    # Build the hover text
                    try:
                        x_val = x_values.iloc[ind_idx] if hasattr(x_values, 'iloc') else x_values[ind_idx]
                        y_val = y_values.iloc[ind_idx] if hasattr(y_values, 'iloc') else y_values[ind_idx]

                        # Format the hover text with X and Y values
                        hover_text = f"X: {x_val:.3f}\nZ: {y_val:.3f}"

                        # Add N value if available
                        if n_values is not None:
                            n_val = n_values.iloc[ind_idx] if hasattr(n_values, 'iloc') else n_values[ind_idx]
                            hover_text += f"\nN: {n_val:.3f}"

                        annot.set_text(hover_text)
                        annot.set_visible(True)
                        self.canvas.draw_idle()
                    except (IndexError, KeyError) as e:
                        self.show_status_message(f"Hover annotation error: {e}", 3000)
                elif vis:
                    annot.set_visible(False)
                    self.canvas.draw_idle()

        # Connect the hover event to the matplotlib figure
        self._hover_cid = self.canvas.mpl_connect("motion_notify_event", hover)
            
    def on_plot_click(self, event):
        """Handle plot click events"""
        # Left click
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            # Check for modifiers
            modifiers = QApplication.keyboardModifiers()
            
            # Remove previous interpolated rows before finding nearest point
            if modifiers == Qt.KeyboardModifier.ControlModifier and self.interpolated_left_idx is not None:
                self.current_data = self.current_data.drop(index=self.interpolated_left_idx, errors='ignore')
                self.interpolated_left_idx = None
                self.update_table()
            elif modifiers == Qt.KeyboardModifier.AltModifier and self.interpolated_right_idx is not None:
                self.current_data = self.current_data.drop(index=self.interpolated_right_idx, errors='ignore')
                self.interpolated_right_idx = None
                self.update_table()
            
            # Find the nearest point
            nearest_idx = self.find_nearest_point(event.xdata, event.ydata)

            if nearest_idx is not None:
                # Get the x value of the nearest point
                x_value = self.current_data.iloc[nearest_idx][self.x_column]

                if modifiers == Qt.KeyboardModifier.ControlModifier:
                    # Set left bank
                    self.set_left_bank(x_value, nearest_idx)
                    self.show_status_message(f"Left bank set at X={x_value}", 1000)
                elif modifiers == Qt.KeyboardModifier.AltModifier:
                    # Set right bank
                    self.set_right_bank(x_value, nearest_idx)
                    self.show_status_message(f"Right bank set at X={x_value}", 1000)

        # Right click
        elif event.button == 3 and event.xdata is not None:
            self.show_plot_context_menu(event)

    def show_plot_context_menu(self, event):
        """Show right click context menu"""
        menu = QMenu()
        add_left_action = menu.addAction("Set LB")
        add_right_action = menu.addAction("Set RB")

        action = menu.exec(QCursor.pos())

        if action == add_left_action:
            self.interpolate_and_set_bank(event.xdata, bank='left')
        elif action == add_right_action:
            self.interpolate_and_set_bank(event.xdata, bank='right')

    def interpolate_and_set_bank(self, x_value, bank='left'):
        """Insert an interpolated row at x_value and set as left/right bank"""
        df = self.current_data
        x_col = self.x_column
        if df is None or x_col not in df.columns:
            return
        
        # Find bounding rows
        prev_row = df[df[x_col] <= x_value].tail(1)
        next_row = df[df[x_col] >= x_value].head(1)
        if prev_row.empty or next_row.empty:
            return  # x_value is out of bounds
        
        prev_idx = prev_row.index[0]
        next_idx = next_row.index[0]
        
        if prev_idx == next_idx:
            return  # x matches an existing point
        
        # Remove previous interpolated bank row if exists
        if bank == 'left' and self.interpolated_left_idx is not None:
            self.current_data = self.current_data.drop(index=self.interpolated_left_idx, errors='ignore')
            # Reset df reference after potential row removal
            df = self.current_data
        elif bank == 'right' and self.interpolated_right_idx is not None:
            self.current_data = self.current_data.drop(index=self.interpolated_right_idx, errors='ignore')
            # Reset df reference after potential row removal
            df = self.current_data
        
        # Find the insertion position in the current dataframe
        insert_position = None
        for i in range(len(df)):
            if df.iloc[i][x_col] > x_value:
                insert_position = i
                break
        
        if insert_position is None:
            insert_position = len(df)  # Insert at end
        
        # Get the bounding rows for interpolation (may have changed after deletion)
        prev_row = df[df[x_col] <= x_value].tail(1)
        next_row = df[df[x_col] >= x_value].head(1)
        
        if prev_row.empty or next_row.empty:
            return  # x_value is out of bounds after deletion
        
        row1 = prev_row.iloc[0]
        row2 = next_row.iloc[0]
        
        # Interpolation ratio — shared by all columns
        x1, x2 = row1[x_col], row2[x_col]
        ratio = (x_value - x1) / (x2 - x1)

        # Create interpolated row
        new_row = {}
        for col in df.columns:
            if col == x_col:
                new_row[col] = x_value
                continue

            col_dtype = df[col].dtype
            v1, v2 = row1[col], row2[col]

            if pd.api.types.is_numeric_dtype(col_dtype) and not pd.api.types.is_bool_dtype(col_dtype):
                # Linear interpolation for numeric columns
                interpolated = v1 + ratio * (v2 - v1)
                # Preserve integer type if both original values are int
                if pd.api.types.is_integer_dtype(col_dtype):
                    interpolated = int(round(interpolated))
                new_row[col] = interpolated
            elif pd.api.types.is_bool_dtype(col_dtype):
                # For booleans: keep if same, else False
                new_row[col] = v1 if v1 == v2 else False
            elif str(col).lower() == 'wkt' and isinstance(v1, str) and isinstance(v2, str):
                # Interpolate WKT POINT coordinates
                m1 = _WKT_POINT_RE.match(v1.strip())
                m2 = _WKT_POINT_RE.match(v2.strip())
                if m1 and m2:
                    coords1 = [float(c) for c in m1.groups() if c is not None]
                    coords2 = [float(c) for c in m2.groups() if c is not None]
                    if len(coords1) == len(coords2):
                        ic = [c1 + ratio * (c2 - c1) for c1, c2 in zip(coords1, coords2)]
                        if len(ic) == 3:
                            new_row[col] = f"POINT Z ({ic[0]:.4f} {ic[1]:.4f} {ic[2]:.4f})"
                        else:
                            new_row[col] = f"POINT ({ic[0]:.4f} {ic[1]:.4f})"
                    else:
                        new_row[col] = np.nan
                else:
                    new_row[col] = np.nan
            elif isinstance(v1, str) and v1 == v2:
                # Copy string if both are identical
                new_row[col] = v1
            else:
                # Mismatched or unsupported types
                new_row[col] = np.nan
        
        # Create new dataframe with the interpolated row inserted
        df_above = df.iloc[:insert_position].copy()
        df_below = df.iloc[insert_position:].copy()
        new_row_df = pd.DataFrame([new_row])
        
        # Concatenate and reset index to get clean sequential indexing
        self.current_data = pd.concat([df_above, new_row_df, df_below], ignore_index=True)
        
        self.update_table()
        
        # Find new row position by x_value (should be at insert_position)
        new_pos = insert_position
        
        if bank == 'left':
            self.set_left_bank(x_value, new_pos)
            self.interpolated_left_idx = new_pos
            self.show_status_message(f"Left bank set at X={x_value}", 1000)
        elif bank == 'right':
            self.set_right_bank(x_value, new_pos)
            self.interpolated_right_idx = new_pos
            self.show_status_message(f"Right bank set at X={x_value}", 1000)
        
    def find_nearest_point(self, x_click, y_click):
        """Find the index of the nearest point to the clicked location, accounting for axis scaling"""
        if self.current_data is None or self.x_column is None or self.y_column is None:
            return None

        x_values = self.current_data[self.x_column].values
        y_values = self.current_data[self.y_column].values

        # Get the axes range for normalization
        x_min, x_max = self.canvas.axes.get_xlim()
        y_min, y_max = self.canvas.axes.get_ylim()

        # Avoid division by zero
        x_range = max(x_max - x_min, 1e-10)
        y_range = max(y_max - y_min, 1e-10)

        x_norm = (x_values - x_min) / x_range
        y_norm = (y_values - y_min) / y_range
        x_click_norm = (x_click - x_min) / x_range
        y_click_norm = (y_click - y_min) / y_range

        # Calculate distances
        distances = np.sqrt((x_norm - x_click_norm)**2 + (y_norm - y_click_norm)**2)

        # Find the index of the minimum distance
        nearest_idx = np.argmin(distances)

        return nearest_idx

    def set_left_bank(self, x_value, index=None):
        """Set the left bank at the given X value"""
        if DEBUG: print(f"[DEBUG set_left_bank] x_value={x_value}, index={index}", flush=True)
        self.left_bank = x_value
        self.left_bank_index = index
        if DEBUG: print("[DEBUG set_left_bank] calling update_plot()", flush=True)
        self.update_plot(preserve_view=True)
        if DEBUG: print("[DEBUG set_left_bank] calling update_cut_indices()", flush=True)
        self.update_cut_indices()
        if DEBUG: print("[DEBUG set_left_bank] done", flush=True)

    def set_right_bank(self, x_value, index=None):
        """Set the right bank at the given X value"""
        if DEBUG: print(f"[DEBUG set_right_bank] x_value={x_value}, index={index}", flush=True)
        self.right_bank = x_value
        self.right_bank_index = index
        if DEBUG: print("[DEBUG set_right_bank] calling update_plot()", flush=True)
        self.update_plot(preserve_view=True)
        if DEBUG: print("[DEBUG set_right_bank] calling update_cut_indices()", flush=True)
        self.update_cut_indices()
        if DEBUG: print("[DEBUG set_right_bank] done", flush=True)

    def apply_banks(self):
        """Apply bank boundaries and return (in_bank_df, out_of_bank_df).

        Both dataframes share the same columns as current_data (no Trim column).
        make_leftmost_zero shifts so the leftmost IN-BANK point equals X=0; the
        same offset is applied to the out-of-bank rows so all data stays in the
        same coordinate space (left-bank trim rows will have negative X).
        """
        if DEBUG: print(f"[DEBUG apply_banks] left_bank={self.left_bank}, right_bank={self.right_bank}", flush=True)

        empty = pd.DataFrame(columns=self.current_data.columns) if self.current_data is not None else pd.DataFrame()

        if self.current_data is None:
            if DEBUG: print("[DEBUG apply_banks] no current_data, returning empty", flush=True)
            return empty, empty

        df = self.current_data.copy()

        # Strip legacy '!# ' markers from the first column (old-format files loaded
        # before this change will have them; clean on the way through).
        first_col = df.columns[0]
        if not pd.api.types.is_numeric_dtype(df[first_col].dtype):
            _pattern = re.compile(r'^\s*[!#]{1,2}\s*')
            df[first_col] = df[first_col].apply(
                lambda v: _pattern.sub('', str(v)) if pd.notna(v) else v
            )

        # Drop legacy Trim column if present
        if 'Trim' in df.columns:
            df = df.drop(columns=['Trim'])

        if self.left_bank is None and self.right_bank is None:
            if DEBUG: print("[DEBUG apply_banks] no banks set, applying make_leftmost_zero to full data if needed", flush=True)
            if self.make_leftmost_zero_check.isChecked():
                df = self.make_leftmost_zero(df)
            return df, pd.DataFrame(columns=df.columns)

        # Build outside mask
        x_values = pd.to_numeric(df[self.x_column], errors='coerce')
        if DEBUG: print(f"[DEBUG apply_banks] x_values dtype={x_values.dtype}, NaN count={x_values.isna().sum()}", flush=True)
        outside_mask = pd.Series(False, index=df.index)

        if self.left_bank is not None:
            try:
                lb_value = float(self.left_bank)
                outside_mask |= x_values < lb_value
                if DEBUG: print(f"[DEBUG apply_banks] left mask applied, lb={lb_value}, rows masked={outside_mask.sum()}", flush=True)
            except (ValueError, TypeError):
                if DEBUG: print(f"[DEBUG apply_banks] left_bank float() failed: {self.left_bank!r}", flush=True)

        if self.right_bank is not None:
            try:
                rb_value = float(self.right_bank)
                outside_mask |= x_values > rb_value
                if DEBUG: print(f"[DEBUG apply_banks] right mask applied, rb={rb_value}, rows masked={outside_mask.sum()}", flush=True)
            except (ValueError, TypeError):
                if DEBUG: print(f"[DEBUG apply_banks] right_bank float() failed: {self.right_bank!r}", flush=True)

        in_bank_df = df[~outside_mask].copy()
        out_of_bank_df = df[outside_mask].copy()

        if DEBUG: print(f"[DEBUG apply_banks] in_bank={len(in_bank_df)} rows, out_of_bank={len(out_of_bank_df)} rows", flush=True)

        # make_leftmost_zero: offset is the minimum X of the IN-BANK rows so that
        # the active cross-section starts at zero (not the full extent).
        if self.make_leftmost_zero_check.isChecked() and not in_bank_df.empty:
            if DEBUG: print("[DEBUG apply_banks] applying make_leftmost_zero using in-bank min", flush=True)
            offset = pd.to_numeric(in_bank_df[self.x_column], errors='coerce').min()
            in_bank_df[self.x_column] = pd.to_numeric(in_bank_df[self.x_column], errors='coerce') - offset
            if not out_of_bank_df.empty:
                out_of_bank_df[self.x_column] = pd.to_numeric(out_of_bank_df[self.x_column], errors='coerce') - offset

        if DEBUG: print(f"[DEBUG apply_banks] done, returning in_bank shape={in_bank_df.shape}, out_of_bank shape={out_of_bank_df.shape}", flush=True)
        return in_bank_df, out_of_bank_df

    def fix_verticals(self, df):
        """Fix vertical values by ensuring no duplicate X values"""
        if df.empty:
            return df

        if self.x_column not in self.x_column_unsortable_preferences:
            prev_x = df.loc[0, self.x_column]

            for i in range(1, len(df)):
                if df.loc[i, self.x_column] <= prev_x:
                    df.loc[i, self.x_column] = prev_x + 0.001
                prev_x = df.loc[i, self.x_column]

        return df

    def make_leftmost_zero(self, df):
        """Make the left most active value equal X=0"""
        if df.empty:
            return df

        min_x = df[self.x_column].min()
        df[self.x_column] = df[self.x_column] - min_x
        # print(f"min_x: {min_x}")

        return df

    def save_file(self):
        """Save the current file with the applied changes"""
        if DEBUG: print("[DEBUG save_file] entered", flush=True)
        self._saving = True
        if self.current_data is None or self.current_file_index < 0:
            if DEBUG: print("[DEBUG save_file] early return - no data", flush=True)
            self._saving = False
            return

        # Get in-bank and out-of-bank data
        if DEBUG: print("[DEBUG save_file] calling apply_banks()", flush=True)
        in_bank_df, out_of_bank_df = self.apply_banks()
        if DEBUG: print(f"[DEBUG save_file] apply_banks done, in_bank shape={in_bank_df.shape}, out_of_bank shape={out_of_bank_df.shape}", flush=True)

        # Determine the output filename
        input_path = self.csv_files[self.current_file_index]
        filename = os.path.basename(input_path)
        directory = os.path.dirname(input_path)

        if self.version_combo.currentText() == "Increment Version":
            # Extract the base name and extension
            base, ext = os.path.splitext(filename)

            # Remove existing version if it has it
            base = re.sub(self.version_regex or r'_(v[\d]+)', '', base)

            # Add new version
            version = self.version_edit.text()
            new_filename = f"{base}_{version}{ext}"
            output_path = os.path.join(directory, new_filename)
        else:
            # Change in-place
            output_path = input_path

        trim_path = os.path.splitext(output_path)[0] + '.trim.csv'
        if DEBUG: print(f"[DEBUG save_file] output_path={output_path}, trim_path={trim_path}", flush=True)

        try:
            if DEBUG: print("[DEBUG save_file] writing main CSV", flush=True)
            in_bank_df.to_csv(output_path, index=False)
            if DEBUG: print("[DEBUG save_file] main CSV written", flush=True)

            # Write trim sidecar if there are out-of-bank rows; otherwise remove stale sidecar
            if not out_of_bank_df.empty:
                if DEBUG: print(f"[DEBUG save_file] writing trim sidecar ({len(out_of_bank_df)} rows)", flush=True)
                out_of_bank_df.to_csv(trim_path, index=False)
                if DEBUG: print("[DEBUG save_file] trim sidecar written", flush=True)
            elif os.path.exists(trim_path):
                if DEBUG: print("[DEBUG save_file] removing stale trim sidecar", flush=True)
                os.remove(trim_path)

            if output_path not in self.all_csv_files:
                self.all_csv_files.append(output_path)
            self.recompute_version_links()
            self.load_current_file()

            # Save plot if requested
            if self.make_plot_file_check.isChecked():
                if DEBUG: print("[DEBUG save_file] calling savefig()", flush=True)
                plot_path = os.path.splitext(output_path)[0] + ".png"
                self.canvas.fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                if DEBUG: print("[DEBUG save_file] savefig done", flush=True)

            if DEBUG: print("[DEBUG save_file] save complete", flush=True)
            self.show_status_message(f"File saved to: {output_path}", 2000)

        except Exception as e:
            if DEBUG: print(f"[DEBUG save_file] exception: {e}", flush=True)
            QMessageBox.critical(self, "Error", f"Error saving file: {str(e)}")
        finally:
            self._saving = False
            if DEBUG: print("[DEBUG save_file] _saving flag cleared", flush=True)

    def on_file_selected(self):
        """Handle selection of a file from the list"""
        if DEBUG: print(f"[DEBUG on_file_selected] called, autosave={self.autosave_check.isChecked()}, _saving={getattr(self, '_saving', False)}", flush=True)
        if getattr(self, '_saving', False):
            if DEBUG: print("[DEBUG on_file_selected] re-entry blocked during save", flush=True)
            return
        if self.autosave_check.isChecked() and self.current_data is not None:
            if DEBUG: print("[DEBUG on_file_selected] autosave triggered save_file()", flush=True)
            self.save_file()

        self.current_file_index = self.file_list_widget.currentRow()
        if DEBUG: print(f"[DEBUG on_file_selected] loading file index={self.current_file_index}", flush=True)
        self.load_current_file()

    def previous_file(self):
        """Load the previous file in the list"""
        if self.csv_files and self.current_file_index > 0:
            # Check if we need to save current changes
            if self.autosave_check.isChecked():
                self.save_file()

            self.current_file_index -= 1
            self.load_current_file()

    def next_file(self):
        """Load the next file in the list"""
        if self.csv_files and self.current_file_index < len(self.csv_files) - 1:
            # Check if we need to save current changes
            if self.autosave_check.isChecked():
                self.save_file()

            self.current_file_index += 1
            self.load_current_file()

    def undo(self):
        """Placeholder for undo functionality"""
        # To be implemented in a future version
        pass

    def redo(self):
        """Placeholder for redo functionality"""
        # To be implemented in a future version
        pass

    def refresh_polygon_layer_combo(self, *_):
        """Repopulate the polygon layer combo with current QGIS project polygon layers."""
        # Remember which layer is currently selected so we can restore it
        current_id = None
        if self.polygon_layer and self.polygon_layer.isValid():
            current_id = self.polygon_layer.id()

        self.polygon_layer_combo.blockSignals(True)
        self.polygon_layer_combo.clear()
        self.polygon_layer_combo.addItem("--- 2d_code_inactive ---", None)

        project = QgsProject.instance()
        assert project is not None
        for layer in project.mapLayers().values():
            if isinstance(layer, QgsVectorLayer) and layer.geometryType() == QgsWkbTypes.PolygonGeometry:
                self.polygon_layer_combo.addItem(layer.name(), layer)

        # Restore previous selection if the layer still exists
        restore_idx = 0
        if current_id:
            for i in range(self.polygon_layer_combo.count()):
                lyr = self.polygon_layer_combo.itemData(i)
                if lyr is not None and lyr.id() == current_id:
                    restore_idx = i
                    break
            else:
                # Layer was removed — clear the stored reference
                self.polygon_layer = None

        # setCurrentIndex before unblocking so the signal doesn't fire during refresh
        # (avoids a recursive loop via layersAdded → refresh → on_polygon_layer_changed → addMapLayer → layersAdded)
        self.polygon_layer_combo.setCurrentIndex(restore_idx)
        self.polygon_layer_combo.blockSignals(False)

    def on_polygon_layer_changed(self, index):
        """Handle polygon layer combo selection change."""
        layer = self.polygon_layer_combo.itemData(index)
        self.polygon_layer = layer if (layer and layer.isValid()) else None

        if self.polygon_layer and self.current_data is not None:
            if self.polygon_layer.isValid():
                self.points_to_path()

    def points_to_path(self):
        """Converts a point CSV file to a path layer"""
        if self.paths_layer and self.paths_layer.isValid():
            QgsProject.instance().removeMapLayer(self.paths_layer)

        self.paths_layer = None

        try:
            # Detect if WKT column exists
            with open(self.file_path, 'r', encoding='utf-8') as f:
                skiplines = 0
                for line in f:
                    if line.strip().startswith('!'):
                        skiplines += 1
                    if line.strip() and not line.strip().startswith('!'):
                        header = line.strip().split(',')
                        break

            header_lc = [h.strip().lower() for h in header]
            header_map = {h.strip().lower(): h.strip() for h in header}  # lowercase -> original case

            has_wkt = 'wkt' in header_lc
            if has_wkt:
                # Load using WKT geometry
                wkt_col = header_map['wkt']
                uri = f"{Path(self.file_path).as_uri()}?type=csv&geometrytype=Point&skipLines={skiplines}&wktField={wkt_col}"
            elif any(col.lower() in header_lc for col in self.easting_column_preferences) and any(col.lower() in header_lc for col in self.northing_column_preferences):
                # Find matching column names using case-insensitive match
                easting_col_lc = next((col.lower() for col in self.easting_column_preferences if col.lower() in header_lc), None)
                northing_col_lc = next((col.lower() for col in self.northing_column_preferences if col.lower() in header_lc), None)

                # Use the original case from header
                easting_col = header_map[easting_col_lc]
                northing_col = header_map[northing_col_lc]

                uri = f"{Path(self.file_path).as_uri()}?type=csv&xField={easting_col}&yField={northing_col}&geometrytype=Point&skipLines={skiplines}"
            else:
                raise Exception(f"CSV must contain either a WKT column or both {self.easting_column_preferences} and {self.northing_column_preferences} fields.")

            point_layer = QgsVectorLayer(uri, self.file_name, "delimitedtext")

            if not point_layer.isValid():
                raise Exception(f"Invalid point layer: {self.file_name}, URI: {uri}")

            result = processing.run(
                "native:pointstopath",
                {
                    'INPUT': point_layer,
                    'CLOSE_PATH': False,
                    'ORDER_EXPRESSION': f'to_real("{self.x_column}")',
                    'NATURAL_SORT': False,
                    'GROUP_EXPRESSION': '',
                    'OUTPUT': 'TEMPORARY_OUTPUT'
                }
            )
            self.num_paths = result.get('NUM_PATHS', None)
            self.paths_layer = result.get('OUTPUT', None)
            self.paths_layer.setName(self.file_name)
            self.paths_layer.loadNamedStyle(self.paths_style)

            # Get the extent of the new layer
            extent = self.paths_layer.extent()

            # Validate the extent (width > 0 or height > 0)
            if extent.width() > 0 or extent.height() > 0:
                # Calculate the margin (200% margin)
                margin_factor = 2.0  # 200%
                width = extent.width()
                height = extent.height()

                # Create a new extent with the margin applied
                margin_width = width * margin_factor
                margin_height = height * margin_factor
                center_x = extent.center().x()
                center_y = extent.center().y()

                new_extent = QgsRectangle(
                    center_x - margin_width / 2,
                    center_y - margin_height / 2,
                    center_x + margin_width / 2,
                    center_y + margin_height / 2
                )

                # Set the canvas extent with the new margin
                canvas = self.iface.mapCanvas()
                canvas.setExtent(new_extent)
                canvas.refresh()
            else:
                raise Exception("Invalid extent: Unable to zoom.")

            # Check if the layer with the given ID is loaded
            layers = QgsProject.instance().mapLayers()
            if self.paths_layer.id() not in layers:
                QgsProject.instance().addMapLayer(self.paths_layer)

        except Exception as e:
            self.show_status_message(f"Error: Unable to convert WKT CSV to paths: {e}", 3000)
            return None, None

    def find_overlap_with_polygon(self):
        """Finds overlap between paths and a polygon layer using QGIS expressions."""
        if not self.paths_layer or not self.polygon_layer:
            self.show_status_message(f"Error: Invalid input layers", 3000)
            return None

        expr = f"""
        to_json(
            array_foreach(
                array_foreach(
                    array_foreach(
                        array_foreach(
                            array_foreach(
                                array_foreach(
                                    overlay_intersects(
                                        '{self.polygon_layer.id()}',
                                        None,
                                        return_details := true
                                    ),
                                    get_feature_by_id(
                                        '{self.polygon_layer.id()}',
                                        @element['id']
                                    )
                                ),
                                geometry(@element)
                            ),
                            intersection(@geometry, @element)
                        ),
                        geometries_to_array(@element)
                    ),
                    array_foreach(
                        @element,
                        array(
                            start_point(@element),
                            end_point(@element)
                        )
                    )
                ),
                array_foreach(
                    @element,
                    map(
                        'in', line_locate_point(@geometry, @element[0]),
                        'out', line_locate_point(@geometry, @element[-1])
                    )
                )
            )
        )
        """
        expression = QgsExpression(expr)
        if expression.hasParserError():
            self.show_status_message(f"expression.parserErrorString(): {expression.parserErrorString()}", 3000)

        context = QgsExpressionContext()
        context.appendScopes(QgsExpressionContextUtils.globalProjectLayerScopes(self.paths_layer))

        results = []
        for feature in self.paths_layer.getFeatures():
            context.setFeature(feature)
            result = expression.evaluate(context)
            if expression.hasEvalError():
                self.show_status_message(f"expression.evalErrorString(): {expression.evalErrorString()}")
            results.append(json.loads(result))

        # print(f"results: {results}")

        # Flatten the list and extract the 'in' and 'out' pairs
        self.overlaps = []
        for feature_result in results:
            for sublist in feature_result:
                for pair in sublist:
                    in_value = pair['in']
                    out_value = pair['out']

                    # Ensure in_value < out_value, swap if necessary
                    if in_value > out_value:
                        in_value, out_value = out_value, in_value

                    self.overlaps.append((in_value, out_value))

        # print(f"self.overlaps: {self.overlaps}")
