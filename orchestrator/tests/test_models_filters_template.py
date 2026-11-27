"""Template tests for models filter partials."""

from pathlib import Path


def test_models_filters_get_form_has_single_fits_mac_control() -> None:
    template_path = (
        Path(__file__).resolve().parents[1]
        / "templates/orchestrator/partials/models_filters.html"
    )
    text = template_path.read_text(encoding="utf-8")
    form_start = text.index('<form method="GET"')
    form_end = text.index("</form>", form_start)
    filter_form = text[form_start:form_end]
    assert filter_form.count('name="fits_mac"') == 1