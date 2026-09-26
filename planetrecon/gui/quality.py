"""Capture quality and the exact preprocessing mask used by the stacker."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QToolTip, QVBoxLayout, QWidget

from planetrecon.pipeline.preprocess import FrameSelection, best_frame_mask, quality_range


class QualityPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.selection = None
        self.selected = np.zeros(0, dtype=bool)
        self.order = np.zeros(0, dtype=int)
        self.cutoff = None
        self.enabled = True
        self.percent = 100
        self.mode = 'quality_range'
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
        self.canvas = QualityCanvas(self)
        layout.addWidget(self.canvas, 1)
        self.legend = QLabel('Green: selected · Orange: below selection · Red: screened out · Bottom strip: mask')
        self.legend.setWordWrap(True)
        layout.addWidget(self.legend)
        self.set_report({}, 100, 'quality_range', True)

    def set_report(self, report, percent, mode, enabled):
        data = report.get('frame_quality') if report.get('status') == 'ready' else None
        if data is not self.data:
            self.data = data
            self.selection = None
            if data is not None:
                scores = np.asarray(data['scores'], dtype=float)
                accepted = np.asarray(data['accepted'], dtype=bool)
                self.selection = FrameSelection(accepted, scores[:, None], {})
                self.low, self.high = quality_range(self.selection)
                self.normalized = np.full(scores.shape, np.nan)
                finite = np.isfinite(scores)
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
        self.canvas.update()

    def _set_order(self, *_):
        if self.selection is not None:
            scores = self.selection.measurements[:, 0]
            self.order = (np.argsort(-np.where(np.isfinite(scores), scores, -np.inf), kind='stable')
                          if self.order_control.currentIndex() else np.arange(len(scores)))
        self.canvas.update()

    def frame_description(self, index):
        score = self.selection.measurements[index, 0]
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
        return f'Frame {index + 1} · Quality {quality}\n{state}' + (': ' + ', '.join(reasons) if reasons else '')


class QualityCanvas(QWidget):
    colors = ('#49c590', '#edae49', '#f07178')

    def __init__(self, plot):
        super().__init__(plot)
        self.plot = plot
        self.setMinimumHeight(160)
        self.setMouseTracking(True)
        self.setAccessibleName('Frame quality graph and exclusion mask')

    def plot_rect(self):
        return QRectF(48, 22, max(1, self.width() - 66), max(1, self.height() - 66))

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
                         'Quality range (%) — worst to best' if plot.low != plot.high else 'Quality range (%) — all measured scores equal')
        for value in (0, 25, 50, 75, 100):
            y = rect.bottom() - value / 100 * rect.height()
            painter.setPen(QColor('#33414f'))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.setPen(QColor('#cbd5df'))
            painter.drawText(QRectF(1, y-9, 40, 18), Qt.AlignmentFlag.AlignRight, str(value))

        order = plot.order
        n = len(order)
        # Aggregate only within display columns. Preserve each class's extrema
        # and mask marks, so a single rejected frame never disappears in a mean.
        columns = max(1, int(rect.width()))
        xbins = np.minimum(columns-1, (np.arange(n) * columns / max(n-1, 1)).astype(int))
        scores = plot.normalized[order]
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
                y1, y2 = (rect.bottom() - value / 100 * rect.height() for value in (lo[col], hi[col]))
                painter.drawLine(QPointF(x, y1), QPointF(x, y2))
                painter.drawPoint(QPointF(x, y1))
            for col in np.unique(xbins[member]):
                x = rect.left() + col * rect.width() / max(columns-1, 1)
                # Separate lanes keep overlapping mask classes visible.
                y = rect.bottom() + 4 + state * 3
                painter.drawLine(QPointF(x, y), QPointF(x, y+2))
        painter.restore()
        if plot.cutoff is not None:
            y = rect.bottom() - (100 - plot.percent) / 100 * rect.height()
            painter.setPen(QPen(QColor('#f4e285'), 1.5, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.drawText(QRectF(rect.right()-190, y-18, 185, 17), Qt.AlignmentFlag.AlignRight,
                             f'Upper {plot.percent}% cutoff')
        painter.setPen(QColor('#cbd5df'))
        bottom = QRectF(rect.left(), rect.bottom()+16, rect.width(), 20)
        painter.drawText(bottom, Qt.AlignmentFlag.AlignLeft, '1')
        painter.drawText(bottom, Qt.AlignmentFlag.AlignRight, f'{n:,}')
        painter.drawText(bottom, Qt.AlignmentFlag.AlignCenter,
                         'Quality rank (best first)' if plot.order_control.currentIndex() else 'Frame number (capture order)')

    def mouseMoveEvent(self, event):
        rect = self.plot_rect()
        if self.plot.selection is None or not len(self.plot.order) or not rect.left() <= event.position().x() <= rect.right():
            QToolTip.hideText()
            return
        rank = round((event.position().x() - rect.left()) / rect.width() * (len(self.plot.order)-1))
        index = int(self.plot.order[rank])
        QToolTip.showText(event.globalPosition().toPoint(), self.plot.frame_description(index), self)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)
