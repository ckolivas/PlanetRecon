"""Manual, reversible RGB alignment previews against an owned original result."""
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QVBoxLayout


def edit_rgb_alignment(parent, offsets, apply):
    dialog = QDialog(parent)
    dialog.setWindowTitle('Align RGB channels')
    layout = QVBoxLayout(dialog)
    description = QLabel('Green stays fixed. Positive X moves colour right; positive Y moves it down.\n'
                         'Offsets are in full-resolution pixels. Apply previews the change without restacking.\n'
                         'Each preview uses the original stack. No sharpening is applied.')
    description.setWordWrap(True)
    layout.addWidget(description)
    form = QFormLayout()
    edits = []
    for colour, pair in zip(('Red', 'Blue'), offsets):
        row = []
        for axis, value in zip(('X', 'Y'), pair):
            edit = QDoubleSpinBox()
            edit.setObjectName(f'{colour.lower()}_{axis.lower()}')
            edit.setRange(-16, 16)
            edit.setDecimals(2)
            edit.setSingleStep(.1)
            edit.setValue(value)
            edit.setToolTip(f'Move {colour.lower()} content relative to green in full-resolution pixels; '
                            f'positive values move it {"right" if axis == "X" else "down"}.')
            form.addRow(f'{colour} {axis}', edit)
            row.append(edit)
        edits.append(row)
    layout.addLayout(form)
    error = QLabel()
    error.setWordWrap(True)
    layout.addWidget(error)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Apply |
                               QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Reset)
    layout.addWidget(buttons)
    tips = {'Ok': 'Apply these channel offsets and close; Save result exports the adjusted image.',
            'Apply': 'Preview these offsets from the original stack, keeping this dialog open.',
            'Cancel': 'Discard changes from this dialog and restore the image shown before opening it.',
            'Reset': 'Preview the original stack with all channel offsets set to zero.'}
    for name, tip in tips.items():
        buttons.button(getattr(QDialogButtonBox.StandardButton, name)).setToolTip(tip)
    def preview():
        try:
            for row in edits:
                for edit in row:
                    edit.interpretText()
            apply(tuple(tuple(edit.value() for edit in row) for row in edits))
            error.clear()
            return True
        except (ValueError, MemoryError) as exc:
            error.setText(str(exc))
            return False
    def reset():
        for row in edits:
            for edit in row:
                edit.setValue(0.)
        preview()
    def accept():
        if preview():
            dialog.accept()
    buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(preview)
    buttons.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(reset)
    buttons.accepted.connect(accept)
    buttons.rejected.connect(dialog.reject)
    return dialog.exec() == QDialog.DialogCode.Accepted
