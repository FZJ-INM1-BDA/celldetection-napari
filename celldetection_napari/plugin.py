from napari import Viewer
from napari.qt.threading import thread_worker
from qtpy import QtWidgets as W
from qtpy import QtGui as G
from magicgui.widgets import create_widget
import napari
from .cpn import CpnInterface
from skimage.util import img_as_ubyte, invert
from skimage.exposure import adjust_gamma
import celldetection as cd
from celldetection.visualization import images as vim
from . import components as co
import numpy as np
import warnings
from imageio.v2 import imread
from os.path import join, basename, dirname, isdir, isfile
from os import makedirs
import warnings

warnings.filterwarnings("ignore")


def process_img(
        img,
        apply_gamma,
        gamma,
        apply_percentile,
        perc_low,
        perc_high,
        apply_invert
):
    if apply_percentile:
        low, high = np.percentile(img, (perc_low, perc_high))
        img = img_as_ubyte((np.clip(img, low, high) - low) / (high - low))

    if apply_gamma:
        img = adjust_gamma(img, gamma)

    if apply_invert:
        img = invert(img)

    return img


@thread_worker
def _run_batch_processing(
        filenames,
        model_interface,
        apply_gamma,
        gamma,
        apply_percentile,
        perc_low,
        perc_high,
        apply_invert,
        out_directory='.',
        keys=None  # keys of out dictionary; values are written to h5
):
    if keys is None:
        keys = ['labels']
    assert not isfile(out_directory)
    makedirs(out_directory, exist_ok=True)
    for file in filenames:
        img = imread(file)
        yield
        img = process_img(
            img,
            apply_gamma=apply_gamma,
            gamma=gamma,
            apply_invert=apply_invert,
            apply_percentile=apply_percentile,
            perc_high=perc_high,
            perc_low=perc_low,
        )
        yield
        out = model_interface(img)
        yield
        out['img'] = img
        dst = join(out_directory, '.'.join(basename(file).split('.')[:-1] + ['h5']))
        cd.to_h5(dst, **{k: out[k] for k in keys})
        yield


@thread_worker
def _run_cpn(
        model_interface,
        img,
        apply_gamma,
        gamma,
        apply_percentile,
        perc_low,
        perc_high,
        apply_invert
):
    img = process_img(
        img,
        apply_gamma=apply_gamma,
        gamma=gamma,
        apply_invert=apply_invert,
        apply_percentile=apply_percentile,
        perc_high=perc_high,
        perc_low=perc_low,
    )
    out = model_interface(img)
    out['img'] = img
    return out


@thread_worker
def _select_model(model_name):
    try:
        interface = CpnInterface(model_name)
    except Exception as e:
        warnings.warn(str(e))
        return None
    return interface


class CellDetectionWidget(W.QWidget):
    def __init__(self, viewer: Viewer, parent=None):
        super().__init__(parent)

        self.layout = W.QFormLayout()
        self.interface = None

        title_font = G.QFont()
        title_font.setBold(True)

        # Image selection
        label = W.QLabel('Input selection')
        label.setMaximumHeight(20)
        label.setFont(title_font)
        self.layout.addRow(label)
        
        self._image_combo = create_widget(annotation=napari.layers.Image)
        self._image_combo.reset_choices()
        viewer.layers.events.inserted.connect(self._image_combo.reset_choices)
        viewer.layers.events.removed.connect(self._image_combo.reset_choices)
        self._image_combo.changed.connect(self._update_inputs)
        self.layout.addRow(self._image_combo.native)

        # Model selection title
        label = W.QLabel('Model selection')
        label.setMaximumHeight(20)
        label.setFont(title_font)
        self.layout.addRow(label)

        # Model selection
        self.model_selection = co.ModelSelection()
        self.layout.addRow(self.model_selection)
        self.model_selection.combo.currentIndexChanged.connect(
            lambda index: self._select_model(self.model_selection.combo.itemText(index)))

        # Model settings
        self.model_settigns = co.ModelSettings()
        self.layout.addRow(self.model_settigns)

        # Data preprocessing
        self.data_preprocessing = co.DataPreprocessing()
        self.layout.addRow(self.data_preprocessing)

        # Output options
        self.output_options = co.OutputSettings()
        self.layout.addRow(self.output_options)

        # Run, reset buttons
        hbox = W.QHBoxLayout()
        hbox.setSpacing(8)
        reset_button = W.QPushButton("&Reset")
        reset_button.clicked.connect(self._reset_default)
        hbox.addWidget(reset_button)
        self.run_button = co.RunButton(self._run_model)
        self.run_button.setEnabled(True)
        hbox.addWidget(self.run_button)
        self.layout.addRow(hbox)

        # Batch processing
        self.batch_processing = co.BatchProcessing(self._run_batch_processing,
                                                   finished_callback=self._reset_run_enabled,
                                                   errored_callback=self._reset_run_enabled)
        self.layout.addRow(self.batch_processing)

        # Progress bar
        self.progress_bar = W.QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.layout.addRow(self.progress_bar)

        # Set layout
        self.setLayout(self.layout)

        # Misc
        self._viewer = viewer
        self._labels_seed = None
        self._cache = {}

        # Select a model
        current_model = self.model_selection.combo.currentText()
        if current_model is not None and len(current_model):
            self._select_model(current_model)

    def _select_model(self, model_name):
        self._set_run_enabled(False, True)
        worker = _select_model(model_name)
        worker.finished.connect(lambda: self._reset_run_enabled())
        worker.returned.connect(self._update_interface)
        worker.start()

    def _bad_interface(self):
        print("Bad Interface", flush=True)
        self.model_selection.remove_selected()

    def _update_interface(self, interface):
        if interface is None:
            self._bad_interface()
            return

        self.interface = interface
        try:
            model: cd.models.CPN = self.interface.model.model  # TODO
            self._update_model(model)
        except Exception as e:
            warnings.warn(str(e))
            self._bad_interface()

    def _update_model(self, model):
        for k, i in self.__dict__.items():
            if isinstance(i, co.ConfigElement):
                i.update_model(model)

    def _update_inputs(self, text):
        
        img = self._image_combo.value.data

        key = 'data_preprocessing.perc_toggle'
        if img.itemsize > 1:
            v = self.data_preprocessing.perc_toggle.isChecked()
            if key not in self._cache:
                self._cache[key] = v
            if not v:
                self.data_preprocessing.perc_toggle.setChecked(True)
        elif key in self._cache:
            self.data_preprocessing.perc_toggle.setChecked(self._cache.pop(key))

        self._reset_run_enabled()

    def _reset_default(self):
        for k, i in self.__dict__.items():
            if isinstance(i, co.ConfigElement):
                i.reset_default()

    def _show_labels(self, labels, reduce=False, **kwargs):
        if reduce:
            labels = labels.max(-1, keepdims=True)  # TODO: replace
        if labels.ndim == 2:
            labels = labels[..., None]
        for z in range(labels.shape[2]):
            labels_ = labels[..., z]
            self._viewer.add_labels(labels_, **kwargs)

    def _show_contours(self, contours, face_color_opacity=.42):
        colors = cd.random_colors_hsv(len(contours))
        face_color = np.concatenate((colors, np.zeros_like(colors[:, :1]) + face_color_opacity), 1)
        self._viewer.add_shapes([c[..., ::-1] for c in contours], shape_type='polygon', edge_width=0.75,
                                face_color=face_color,
                                edge_color=colors,
                                name='Contours')

    def _run_batch_processing(self, filenames):
        self._set_run_enabled(False, True)
        worker = _run_batch_processing(
            filenames=filenames,
            model_interface=self.interface,
            apply_gamma=self.data_preprocessing.gamma_toggle.isChecked(),
            gamma=self.data_preprocessing.gamma.get(),
            apply_invert=self.data_preprocessing.invert_toggle.isChecked(),
            apply_percentile=self.data_preprocessing.perc_toggle.isChecked(),
            perc_high=self.data_preprocessing.perc_high.get(),
            perc_low=self.data_preprocessing.perc_low.get()
        )
        return worker

    def _run_model(self):
        self.run_button.set_running(True)
        self._set_run_enabled(False, True)

        img = self._image_combo.value.data

        print("Img", img.dtype, img.shape, flush=True)

        self.interface.tile_size = self.model_settigns.tile_size_slider.get()
        self.interface.overlap = self.model_settigns.overlap_slider.get()

        worker = _run_cpn(
            model_interface=self.interface,
            img=img,
            apply_gamma=self.data_preprocessing.gamma_toggle.isChecked(),
            gamma=self.data_preprocessing.gamma.get(),
            apply_invert=self.data_preprocessing.invert_toggle.isChecked(),
            apply_percentile=self.data_preprocessing.perc_toggle.isChecked(),
            perc_high=self.data_preprocessing.perc_high.get(),
            perc_low=self.data_preprocessing.perc_low.get(),
        )
        self.run_button.register_worker(worker)
        worker.finished.connect(self._reset_run_enabled)
        worker.returned.connect(self._show_output)
        worker.start()

    def _set_run_enabled(self, value, update_progress_bar=False):
        for i in (self.run_button, self.batch_processing.run):
            if not i.running:
                i.setEnabled(value)
            if value:
                self.run_button.set_running(False)
        if update_progress_bar:
            self.progress_bar.setVisible(not value)

    def _reset_run_enabled(self):
        self._set_run_enabled(bool(len(self._viewer.layers)))
        self.batch_processing._update_run_enabled()
        self.progress_bar.setVisible(False)

    def _show_output(self, out: dict):
        labels = out['labels']
        img = out['img']

        op = self.output_options

        if op.multi_labels_button.isChecked():
            self._show_labels(labels, reduce=False, name='Multichannel Labels')

        if op.labels_button.isChecked():
            self._show_labels(labels, reduce=True, name='Labels')

        if op.boxes_button.isChecked():
            text = None
            if op.scores_button.isChecked():
                text = vim._score_texts(scores=out['scores'])
            boxes = out['boxes'].reshape((-1, 2, 2))[..., ::-1]
            self._viewer.add_shapes(
                list(boxes),
                face_color='transparent',
                edge_color='#4AF626',
                name='bounding box',
                text=text
            )

        if op.processed_img_toggle.isChecked():
            self._viewer.add_image(img)

        if op.contours_toggle.isChecked():
            self._show_contours(out['viewable_contours'])
