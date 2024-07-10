from qtpy import QtWidgets as W
from typing import List, Tuple, Union, Sequence, Any
from os.path import join, isfile, isdir, basename, dirname
import torch
from qtpy.QtCore import Qt


def set_model_attr_(model, name, val):
    model.__dict__[name] = val


def get_model_attr(model, name):
    return model.__dict__[name]


class Slider(W.QSlider):
    def __init__(self, default, display_name, vmin=None, vmax=None, div=1, step=1, dtype=None):
        super().__init__(Qt.Horizontal)
        self.label = W.QLabel(display_name)
        self._dtype = dtype
        self._div = div
        if None not in (vmin, vmax):
            self.setRange(vmin * self._div, vmax * self._div)
        self._default = None
        self.setSingleStep(step)
        self.setMinimumWidth(192)
        self.display = W.QLabel('')
        self.display.setMinimumWidth(56)
        self.display.setAlignment(Qt.AlignRight)
        self.valueChanged.connect(self.set)
        self.update_default(default)
        self.update_display()

    def setRange(self, p_int, p_int_1):
        super().setRange(int(p_int), int(p_int_1))

    def setSliderPosition(self, p_int):
        super().setSliderPosition(int(p_int))

    def update_default(self, default):
        self._default = default
        if self._default is not None:
            self.reset_default()

    @property
    def current_value(self):
        return self._current_value

    def reset_default(self):
        self.set(self._default * self._div)
        self.setSliderPosition(self._default * self._div)

    def prep_val(self, val):
        val = val / self._div
        if self._dtype is not None:
            val = self._dtype(val)
        return val

    def update_display(self, val=None):
        if val is None:
            val = self.value()
        val = self.prep_val(val)
        self.display.setText('%0.2f' % val)
        return val

    def get(self):
        return self.prep_val(self.value())

    def set(self, val):
        self.update_display(val)


class ModelSlider(Slider):
    def __init__(self, model, key, display_name, vmin, vmax, div=1, step=1, dtype=None):
        self._key = key
        self._model = None
        super().__init__(display_name=display_name, default=None,
                         vmin=vmin, vmax=vmax, div=div, step=step, dtype=dtype)
        self.update_model(model)

    def update_model(self, model):
        self._model = model
        if model is not None:
            self.update_default(get_model_attr(model, self._key))

    def set(self, val):
        val = self.update_display(val)
        set_model_attr_(self._model, self._key, val)


class ScoreSlider(ModelSlider):
    def __init__(self, model, key='score_thresh', display_name='Score threshold:'):
        super().__init__(model=model, key=key, vmin=0., vmax=1., div=100, display_name=display_name)


class NmsSlider(ModelSlider):
    def __init__(self, model, key='nms_thresh', display_name='NMS threshold:'):
        super().__init__(model=model, key=key, vmin=0., vmax=1., div=100, display_name=display_name)


class SamplesSlider(ModelSlider):
    def __init__(self, model, key='samples', display_name='Sampling points:'):
        super().__init__(model=model, key=key, vmin=4, vmax=512, div=1, dtype=int, display_name=display_name)


class RefinementIterationsSlider(ModelSlider):
    def __init__(self, model, key='refinement_iterations', display_name='Refine iterations:'):
        super().__init__(model=model, key=key, vmin=None, vmax=None, div=1, dtype=int, display_name=display_name)

    def update_default(self, default):
        super().update_default(default)
        if default is not None:
            self.setRange(1, default)
            self.setEnabled(self._default)
            self.setVisible(self._default)


class OrderSlider(ModelSlider):
    def __init__(self, model, key='order', display_name='Order:'):
        super().__init__(model=model, key=key, vmin=None, vmax=None, div=1, display_name=display_name, dtype=int)

    def update_default(self, default):
        super().update_default(default)
        if default is not None:
            self.setRange(1, default)


class PercentileSlider(Slider):
    def __init__(self, default, display_name, vmin=0, vmax=100, div=100, step=1, dtype=None):
        super().__init__(display_name=display_name, default=default, vmin=vmin, vmax=vmax, div=div, step=step,
                         dtype=dtype)


class CheckBox(W.QCheckBox):
    def __init__(self, default=False):
        super().__init__(clicked=lambda val: self.set(val=val))
        self._default = default
        self.reset_default()

    def update_default(self, default):
        self._default = default
        if default is not None:
            self.reset_default()

    def reset_default(self):
        self.setChecked(self._default)

    def set(self, val):
        pass


class RefinementToggle(CheckBox):
    def __init__(self, model, key='refinement'):
        super().__init__()
        self._key = key
        self._model = None
        self.update_model(model)

    def update_default(self, default):
        super().update_default(default)
        if default is not None:
            self.setEnabled(self._default)
            self.setVisible(self._default)

    def update_model(self, model):
        self._model = model
        if model is not None:
            self.update_default(get_model_attr(model, self._key))

    def set(self, val):
        set_model_attr_(self._model, self._key, val)


class FileList(W.QListWidget):
    def __init__(self, removed_callback=None):
        super().__init__()
        self.removed_callback = removed_callback

    @property
    def filenames(self):
        return [self.item(i).text() for i in range(self.count())]

    def remove_selected(self):
        for item in self.selectedItems():
            self.takeItem(self.row(item))

    def remove_all(self):
        for _ in range(self.count()):
            self.takeItem(0)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        key = event.key()
        if key == Qt.Key_Delete or key == Qt.Key_Backspace:
            if event.modifiers() & Qt.ShiftModifier:
                self.remove_all()
            else:
                self.remove_selected()
            if self.removed_callback is not None:
                self.removed_callback()


class RunButton(W.QPushButton):
    def __init__(self, run_callback=None, cancel_callback=None, while_running='Cancel', while_idle='Run'):
        self._texts = [while_idle, while_running]
        self.running = None
        self._worker = None
        super().__init__('Run')
        self.clicked.connect(self.on_click)
        self._run_callback = run_callback
        self._cancel_callback = cancel_callback

    def set_running(self, val):
        self.setText(self._texts[int(val)])
        self.running = val

    def register_worker(self, val):
        self._worker = val
        self._worker.finished.connect(lambda: self.set_running(False))
        self._worker.errored.connect(lambda: self.set_running(False))

    def on_click(self):
        if self.running:
            if self._worker is not None:
                self._worker.quit()
                self._worker = None
            if self._cancel_callback is not None:
                self._cancel_callback()
        elif self._run_callback is not None:
            self._run_callback()


class BatchProcessing(W.QGroupBox):
    def __init__(
            self,
            threaded_run_callback,  # must return worker
            finished_callback=None,
            returned_callback=None,
            errored_callback=None,
            input_formats: List[str] = None
    ):
        if input_formats is None:
            input_formats = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
        self.input_formats = input_formats
        super().__init__('Batch processing')
        self.dialog_button = W.QPushButton('Add files')
        self.dialog_button.clicked.connect(self._open_dialog)
        self.list = FileList(self._update_run_enabled)
        self.list.setMinimumHeight(1)
        self.run = RunButton(self._run)
        self.run.setEnabled(False)
        box = W.QVBoxLayout()
        hbox = W.QHBoxLayout()
        hbox.addWidget(self.dialog_button)
        hbox.addWidget(self.run)
        hbox.setSpacing(8)
        box.addWidget(self.list)
        box.addItem(hbox)
        self.setLayout(box)
        self.threaded_run_callback = threaded_run_callback
        self.finished_callback = finished_callback
        self.returned_callback = returned_callback
        self.errored_callback = errored_callback

    def _update_run_enabled(self):
        self.run.setEnabled(bool(self.list.count()))

    def _open_dialog(self):
        dialog = W.QFileDialog(self)
        dialog.setFileMode(W.QFileDialog.FileMode.ExistingFiles)
        dialog.setViewMode(W.QFileDialog.ViewMode.List)
        dialog.setNameFilter(f"Images ({' '.join(self.input_formats)})")
        if dialog.exec():
            filenames = dialog.selectedFiles()
            filenames = list(set(filenames) - set(self.list.filenames))
            self.list.addItems(filenames)
        self._update_run_enabled()

    def _run(self):
        print("BATCH PROCESSING", self.list.filenames)
        self.run.set_running(True)
        worker = self.threaded_run_callback(self.list.filenames)
        self.run.register_worker(worker)
        if self.returned_callback is not None:
            worker.returned.connect(self.returned_callback)
        if self.finished_callback is not None:
            worker.finished.connect(self.finished_callback)
        if self.errored_callback is not None:
            worker.errored.connect(self.errored_callback)
        worker.start()


class Description(W.QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.setFixedWidth(128)


class ConfigElement(W.QGroupBox):
    def __init__(self, *args, fixed_height=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.vbox = W.QVBoxLayout()
        self.vbox.setSpacing(2)
        self.vbox.setAlignment(Qt.AlignTop)
        self.setLayout(self.vbox)
        self._fixed_height = fixed_height

    def add_row_(self, text, mod):
        h = W.QHBoxLayout()
        d = Description(text + ':')
        s = [mod, d]
        h.addWidget(d)
        h.addWidget(mod)
        if hasattr(mod, 'display'):
            di = mod.display
            s += [di]
            h.addWidget(di)

        for i in [mod, d]:
            i.setFixedHeight(self._fixed_height)
        self.vbox.addLayout(h)

    def reset_default(self):
        for k, i in self.__dict__.items():
            if hasattr(i, 'reset_default') and hasattr(i, '_default'):
                i.reset_default()

    def update_model(self, model):
        for k, i in self.__dict__.items():
            if hasattr(i, 'update_model') and hasattr(i, '_default'):
                i.update_model(model)


class DataPreprocessing(ConfigElement):
    def __init__(self):
        super().__init__('Data preprocessing')
        self.perc_toggle = CheckBox()
        self.perc_low = PercentileSlider(0, 'Percentile low', vmin=0, vmax=100)
        self.perc_high = PercentileSlider(100, 'Percentile high', vmin=0, vmax=100)
        self.gamma_toggle = CheckBox()
        self.gamma = Slider(1., 'Gamma', vmin=0.01, vmax=3., div=100)
        self.invert_toggle = CheckBox()

        add_row_ = self.add_row_
        add_row_('Percentile norm', self.perc_toggle)
        add_row_('Percentile low', self.perc_low)
        add_row_('Percentile high', self.perc_high)
        add_row_('Gamma correction', self.gamma_toggle)
        add_row_('Gamma', self.gamma)
        add_row_('Invert image', self.invert_toggle)


class ModelSettings(ConfigElement):
    def __init__(self, model=None):
        super().__init__('Model settings')

        self.score_slider = ScoreSlider(model=model)
        self.nms_slider = NmsSlider(model=model)
        self.order_slider = OrderSlider(model=model)
        self.samples_slider = SamplesSlider(model=model)
        self.refinement_toggle = RefinementToggle(model)
        self.refinement_slider = RefinementIterationsSlider(model=model)
        self.tile_size_slider = Slider(768, 'Tile size:', vmin=256, vmax=4096, div=1 / 128, step=1, dtype=int)
        self.overlap_slider = Slider(384, 'Overlap size:', 64, 2048, div=1 / 32, dtype=int)

        add_row_ = self.add_row_
        add_row_('Score threshold', self.score_slider)
        add_row_('NMS threshold', self.nms_slider)
        add_row_('Order', self.order_slider)
        add_row_('Sampling points', self.samples_slider)
        add_row_('Refinement', self.refinement_toggle)
        add_row_('Refinement steps', self.refinement_slider)
        add_row_('Tile size', self.tile_size_slider)
        add_row_('Overlap size', self.overlap_slider)


class OutputSettings(ConfigElement):
    def __init__(self):
        super().__init__('Output options')
        self.labels_button = W.QCheckBox("Labels", clicked=self._update_widgets)
        self.multi_labels_button = W.QCheckBox("Multichannel Labels", clicked=self._update_widgets)
        self.boxes_button = W.QCheckBox("Bounding Boxes", clicked=self._update_widgets)
        self.scores_button = W.QCheckBox("Scores", clicked=self._update_widgets)
        self.contours_toggle = W.QCheckBox("Contours")
        self.processed_img_toggle = W.QCheckBox("Processed Image")

        self.labels_button.setChecked(True)

        h = W.QHBoxLayout()
        h.addWidget(self.boxes_button)
        h.addWidget(self.scores_button)

        self.vbox.addWidget(self.labels_button)
        self.vbox.addWidget(self.multi_labels_button)
        self.vbox.addLayout(h)
        self.vbox.addWidget(self.contours_toggle)
        self.vbox.addWidget(self.processed_img_toggle)

        self._cache = {}
        self._update_widgets()

    def _update_widgets(self):
        box = self.boxes_button.isChecked()
        sco_en = self.scores_button.isEnabled()
        if not box and sco_en:
            self._cache['_scores_button'] = self.scores_button.isChecked()
            self.scores_button.setEnabled(False)
            self.scores_button.setChecked(False)
        elif box and not sco_en:
            self.scores_button.setEnabled(True)
            self.scores_button.setChecked(self._cache.get('_scores_button', False))

        box_checked = self.boxes_button.isChecked()
        self.scores_button.setEnabled(box_checked)
        self.scores_button.setVisible(box_checked)


class ModelSelection(W.QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.dialog_button = W.QPushButton('+')
        self.dialog_button.setFixedWidth(32)
        self.dialog_button.clicked.connect(self._open_dialog)
        self.combo = W.QComboBox()
        self.combo.addItems([
            'ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c'
        ])

        self.addWidget(self.combo)
        self.addWidget(self.dialog_button)

    def remove_selected(self):
        self.combo.removeItem(self.combo.currentIndex())

    def _open_dialog(self):
        dialog = W.QFileDialog()
        dialog.setFileMode(W.QFileDialog.FileMode.ExistingFile)
        dialog.setViewMode(W.QFileDialog.ViewMode.List)
        dialog.setNameFilter("Checkpoints (*.pt *.pth *.ckpt)")
        hub_dir = join(torch.hub.get_dir(), 'checkpoints')
        if isdir(hub_dir):
            dialog.setDirectory(hub_dir)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            if filenames:
                filename, = filenames
                if isfile(filename):
                    if filename not in [self.combo.itemText(i) for i in range(self.combo.count())]:
                        self.combo.addItem(filename)
                    self.combo.setCurrentText(filename)
