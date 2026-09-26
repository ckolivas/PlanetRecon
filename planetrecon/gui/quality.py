"""Capture quality and the exact preprocessing mask used by the stacker."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QSplitter, QToolTip, QVBoxLayout, QWidget

from planetrecon.pipeline.preprocess import FrameSelection, best_frame_mask, quality_range


class QualityPlot(QWidget):
    frameRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.selection = None
        self.selected = np.zeros(0, dtype=bool)
        self.order = np.zeros(0, dtype=int)
        self.quality_order = np.zeros(0, dtype=int)
        self.quality_ranks = np.zeros(0, dtype=int)
        self.cutoff = None
        self.enabled = True
        self.percent = 100
        self.mode = 'quality_range'
        self.preview_index = None
        self.browsing_enabled = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(2)
        row = QHBoxLayout()
        title = QLabel('Frame quality')
        title.setStyleSheet('font-weight: bold')
        row.addWidget(title)
        self.order_control = QComboBox()
        self.order_control.addItems(['Capture order', 'Quality rank'])
        self.order_control.currentIndexChanged.connect(self._set_order)
        row.addWidget(self.order_control)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        row.addWidget(self.summary, 1)
        layout.addLayout(row)
        axes = QHBoxLayout()
        self.absolute_control = QCheckBox('Absolute quality (from zero)')
        self.absolute_control.setToolTip('Show raw quality scores on an axis starting at zero instead of the capture’s worst-to-best range.')
        self.log_control = QCheckBox('Log scale')
        self.log_control.setToolTip('Logarithmic above 0.1% of the axis maximum; linear near zero so zero-quality frames remain visible.')
        for control in (self.absolute_control, self.log_control):
            control.toggled.connect(lambda *_: self.canvas.update())
            axes.addWidget(control)
        axes.addStretch(1)
        layout.addLayout(axes)
        graph = QWidget()
        graph_layout = QVBoxLayout(graph)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = QualityCanvas(self)
        graph_layout.addWidget(self.canvas, 1)
        self.legend = QLabel('Green: selected · Orange: below selection · Red: screened out · Bottom strip: mask')
        self.legend.setWordWrap(True)
        graph_layout.addWidget(self.legend)
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_label = QLabel('First frame appears when the capture is loaded.')
        self.frame_label.setWordWrap(True)
        preview_layout.addWidget(self.frame_label)
        self.frame_image = FrameImage()
        preview_layout.addWidget(self.frame_image, 1)
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(graph)
        self.splitter.addWidget(preview)
        self.splitter.setSizes([280, 230])
        layout.addWidget(self.splitter, 1)
        self.set_report({}, 100, 'quality_range', True)

    def clear_preview(self):
        self.preview_index = None
        self.frame_image.image = QImage()
        self.frame_image.update()
        self.frame_label.setText('First frame appears when the capture is loaded.')
        self.canvas.update()

    def set_frame_preview(self, index, image):
        self.preview_index = index
        self.frame_image.image = image
        self.frame_image.update()
        self._frame_caption()
        self.canvas.update()

    def loading_frame(self, index):
        self.clear_preview()
        description = (self.frame_description(index).replace('\n', ' · ')
                       if self.selection is not None and index < len(self.selected)
                       else f'Frame {index + 1:,}')
        self.frame_label.setText(f'{description} · Loading preview…')

    def set_browsing_enabled(self, enabled):
        self.browsing_enabled = enabled
        self.canvas.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def _frame_caption(self):
        if self.preview_index is not None:
            index = self.preview_index
            description = (self.frame_description(index) if self.selection is not None
                           and index < len(self.selected) else f'Frame {index + 1}')
            self.frame_label.setText(description.replace('\n', ' · '))

    def set_report(self, report, percent, mode, enabled):
        data = report.get('frame_quality') if report.get('status') == 'ready' else None
        if data is not self.data:
            self.data = data
            self.selection = None
            self.quality_order = np.zeros(0, dtype=int)
            self.quality_ranks = np.zeros(0, dtype=int)
            if data is not None:
                scores = np.asarray(data['scores'], dtype=float)
                accepted = np.asarray(data['accepted'], dtype=bool)
                self.selection = FrameSelection(accepted, scores[:, None], {})
                self.low, self.high = quality_range(self.selection)
                self.normalized = np.full(scores.shape, np.nan)
                finite = np.isfinite(scores)
                # Match the graph exactly: stable ties in capture order and
                # unavailable scores at the end, including screened frames.
                self.quality_order = np.argsort(-np.where(finite, scores, -np.inf), kind='stable')
                self.quality_ranks = np.empty(len(scores), dtype=int)
                self.quality_ranks[self.quality_order] = np.arange(1, len(scores) + 1)
                if self.low is not None:
                    self.normalized[finite] = (100 * (scores[finite] - self.low) / (self.high - self.low)
                                              if self.high > self.low else 100.)
                self._set_order()
        self.percent, self.mode, self.enabled = percent, mode, enabled
        self.cutoff = None
        if self.selection is None:
            self.selected = np.zeros(0, dtype=bool)
            self.order = np.zeros(0, dtype=int)
            self.summary.setText('Open a preprocessed capture or use Preprocess to load frame quality.')
        else:
            self.selected = (best_frame_mask(self.selection, percent, mode) if enabled
                             else np.ones(len(self.selection.accepted), dtype=bool))
            if not enabled:
                self.summary.setText('Cached screening and quality selection disabled')
            else:
                screened = int((~self.selection.accepted).sum())
                below = int((self.selection.accepted & ~self.selected).sum())
                selection_text = f'Best {percent}% by count'
                if mode == 'quality_range':
                    selection_text = f'Upper {percent}% of quality range'
                    if self.low is not None and self.high > self.low:
                        self.cutoff = self.low + (self.high - self.low) * (1 - percent / 100.)
                        selection_text += f' · cutoff {self.cutoff:.5g}'
                    else:
                        selection_text += ' · flat/unavailable range'
                self.summary.setText(f'{selection_text} · {self.selected.sum():,}/{len(self.selected):,} selected'
                                     f' · {screened:,} screened out · {below:,} below selection')
        self.legend.setText(
            'Green: selected · Orange: below selection · Red: screened out · Bottom strip: mask · Before registration rejection'
            if enabled else 'Grey: measured quality · Screening masks are inactive; run-time validity checks still apply')
        self.order_control.setEnabled(self.selection is not None)
        self._frame_caption()
        self.canvas.update()

    def _set_order(self, *_):
        if self.selection is not None:
            self.order = (self.quality_order if self.order_control.currentIndex()
                          else np.arange(len(self.quality_order)))
        self.canvas.update()

    def axis_maximum(self):
        if self.absolute_control.isChecked() and self.selection is not None:
            return self.high if self.high is not None and self.high > 0 else 1.
        return 100.

    def axis_fraction(self, values):
        """Map displayed axis units to height, preserving zero on either scale."""
        fraction = np.asarray(values, dtype=float) / self.axis_maximum()
        if self.log_control.isChecked():
            # Three logarithmic decades above a linear toe. Work in relative
            # units to keep very small absolute quality scores well conditioned.
            relative = fraction * 1000
            fraction = np.where(relative <= 1, relative,
                                1 + np.log10(np.maximum(relative, 1))) / 4
        return fraction

    def quality_position(self, scores):
        values = np.asarray(scores, dtype=float)
        if not self.absolute_control.isChecked():
            values = ((values - self.low) / (self.high - self.low) * 100
                      if self.low is not None and self.high > self.low
                      else np.where(np.isfinite(values), 100., np.nan))
        return self.axis_fraction(values)

    def axis_ticks(self):
        fractions = (0, .001, .01, .1, 1) if self.log_control.isChecked() else (0, .25, .5, .75, 1)
        return np.asarray(fractions) * self.axis_maximum()

    def axis_title(self):
        title = ('Absolute quality' if self.absolute_control.isChecked() else
                 'Quality range (%) — worst to best' if self.low != self.high else
                 'Quality range (%) — all measured scores equal')
        return title + (' · log (linear near zero)' if self.log_control.isChecked() else '')

    def frame_description(self, index):
        score = self.selection.measurements[index, 0]
        rank, total = int(self.quality_ranks[index]), len(self.quality_ranks)
        ranking = f'Rank {rank:,}/{total:,} (top {100 * rank / total:.2f}% of all frames)'
        quality = f'{score:.6g} ({self.normalized[index]:.1f}% of range)' if np.isfinite(score) else 'unavailable'
        reasons = [key.replace('_', ' ') for key, indices in self.data['reasons'].items() if index in indices]
        if not self.enabled:
            state = 'Screening disabled'
        elif not self.selection.accepted[index]:
            state = 'Screened out'
        elif not self.selected[index]:
            state = 'Below selection'
        else:
            state = 'Selected'
        return f'Frame {index + 1:,} · {ranking} · Quality {quality}\n{state}' + (': ' + ', '.join(reasons) if reasons else '')


class QualityCanvas(QWidget):
    colors = ('#49c590', '#edae49', '#f07178')

    def __init__(self, plot):
        super().__init__(plot)
        self.plot = plot
        self.setMinimumHeight(160)
        self.setMouseTracking(True)
        self.setAccessibleName('Frame quality graph and exclusion mask')

    def plot_rect(self):
        ticks = self.plot.axis_ticks()
        gutter = max(48, max(self.fontMetrics().horizontalAdvance(f'{v:.3g}') for v in ticks) + 12)
        return QRectF(gutter, 22, max(1, self.width() - gutter - 18), max(1, self.height() - 66))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor('#17212b'))
        rect = self.plot_rect()
        plot = self.plot
        painter.setPen(QColor('#cbd5df'))
        if plot.selection is None or not len(plot.order):
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'No validated frame quality loaded')
            return
        painter.drawText(QRectF(8, 1, self.width()-16, 20), Qt.AlignmentFlag.AlignLeft,
                         plot.axis_title())
        for value in plot.axis_ticks():
            y = rect.bottom() - plot.axis_fraction(value) * rect.height()
            painter.setPen(QColor('#33414f'))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.setPen(QColor('#cbd5df'))
            painter.drawText(QRectF(1, y-9, rect.left()-9, 18), Qt.AlignmentFlag.AlignRight, f'{value:.3g}')

        order = plot.order
        n = len(order)
        # Aggregate only within display columns. Preserve each class's extrema
        # and mask marks, so a single rejected frame never disappears in a mean.
        columns = max(1, int(rect.width()))
        xbins = np.minimum(columns-1, (np.arange(n) * columns / max(n-1, 1)).astype(int))
        scores = plot.quality_position(plot.selection.measurements[order, 0])
        if plot.enabled:
            states = np.where(~plot.selection.accepted[order], 2, np.where(plot.selected[order], 0, 1))
            colors = self.colors
        else:
            states = np.zeros(n, dtype=int)
            colors = ('#9aabbc',)
        painter.save()
        painter.setClipRect(rect.adjusted(-2, -2, 2, 14))
        for state, color in enumerate(colors):
            member = states == state
            measured = member & np.isfinite(scores)
            lo, hi = np.full(columns, np.inf), np.full(columns, -np.inf)
            np.minimum.at(lo, xbins[measured], scores[measured])
            np.maximum.at(hi, xbins[measured], scores[measured])
            painter.setPen(QPen(QColor(color), 1.5))
            for col in np.flatnonzero(np.isfinite(lo)):
                x = rect.left() + col * rect.width() / max(columns-1, 1)
                y1, y2 = (rect.bottom() - value * rect.height() for value in (lo[col], hi[col]))
                painter.drawLine(QPointF(x, y1), QPointF(x, y2))
                painter.drawPoint(QPointF(x, y1))
            for col in np.unique(xbins[member]):
                x = rect.left() + col * rect.width() / max(columns-1, 1)
                # Separate lanes keep overlapping mask classes visible.
                y = rect.bottom() + 4 + state * 3
                painter.drawLine(QPointF(x, y), QPointF(x, y+2))
        painter.restore()
        if plot.cutoff is not None:
            y = rect.bottom() - plot.quality_position(plot.cutoff) * rect.height()
            painter.setPen(QPen(QColor('#f4e285'), 1.5, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.drawText(QRectF(rect.right()-190, y-18, 185, 17), Qt.AlignmentFlag.AlignRight,
                             f'Upper {plot.percent}% cutoff')
        if plot.preview_index is not None:
            ranks = np.flatnonzero(order == plot.preview_index)
            if len(ranks):
                x = rect.left() + int(ranks[0]) / max(n-1, 1) * rect.width()
                painter.setPen(QPen(QColor('#ffffff'), 1, Qt.PenStyle.DotLine))
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()+12))
                score = plot.quality_position(plot.selection.measurements[plot.preview_index, 0])
                if np.isfinite(score):
                    painter.setBrush(QColor('#ffffff'))
                    painter.drawEllipse(QPointF(x, rect.bottom()-score*rect.height()), 3, 3)
        painter.setPen(QColor('#cbd5df'))
        bottom = QRectF(rect.left(), rect.bottom()+16, rect.width(), 20)
        painter.drawText(bottom, Qt.AlignmentFlag.AlignLeft, '1')
        painter.drawText(bottom, Qt.AlignmentFlag.AlignRight, f'{n:,}')
        painter.drawText(bottom, Qt.AlignmentFlag.AlignCenter,
                         'Quality rank (best first)' if plot.order_control.currentIndex() else 'Frame number (capture order)')

    def frame_at(self, position):
        rect = self.plot_rect()
        if (self.plot.selection is None or not len(self.plot.order)
                or not rect.adjusted(0, 0, 0, 12).contains(position)):
            return None
        rank = round((position.x() - rect.left()) / rect.width() * (len(self.plot.order)-1))
        return int(self.plot.order[rank])

    def mousePressEvent(self, event):
        index = self.frame_at(event.position())
        if event.button() == Qt.MouseButton.LeftButton and self.plot.browsing_enabled and index is not None:
            self.plot.frameRequested.emit(index)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        index = self.frame_at(event.position())
        if index is None:
            QToolTip.hideText()
            return
        QToolTip.showText(event.globalPosition().toPoint(), self.plot.frame_description(index), self)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)


class FrameImage(QWidget):
    """Fit the selected frame without letting its dimensions resize the layout."""
    def __init__(self):
        super().__init__()
        self.image = QImage()
        self.setMinimumHeight(120)
        self.setAccessibleName('Selected capture frame')

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#101820'))
        if not self.image.isNull():
            size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRectF((self.width()-size.width())/2, (self.height()-size.height())/2,
                            size.width(), size.height())
            painter.drawImage(target, self.image)
