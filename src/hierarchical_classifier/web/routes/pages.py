"""Browser routes keep HTML concerns separate from classification logic."""

from flask import Blueprint, Response, current_app, render_template, request
from werkzeug.exceptions import BadRequest

from hierarchical_classifier.services.batch import classify_csv, results_csv
from hierarchical_classifier.web.validation import classifier, text_payload

pages = Blueprint("pages", __name__)


@pages.route("/", methods=["GET", "POST"])
def index():
    """Render a single-input form and an accessible prediction result."""
    title, text, result, error = "", "", None, None
    status = 200
    if request.method == "POST":
        title, text = request.form.get("Title", ""), request.form.get("Text", "")
        try:
            title, text = text_payload({"Title": title, "Text": text})
            result = classifier().predict(title, text)
        except (BadRequest, ValueError) as exc:
            error = exc.description if isinstance(exc, BadRequest) else str(exc)
            status = 400
    return (
        render_template(
            "index.html", title=title, text=text, result=result, error=error
        ),
        status,
    )


@pages.route("/batch", methods=["GET", "POST"])
def batch():
    """Show upload instructions or return a safe downloadable result CSV."""
    error = None
    if request.method == "POST":
        upload = request.files.get("file")
        try:
            if upload is None:
                raise ValueError("Choose a CSV file first.")
            rows = classify_csv(upload.read(), classifier())
            return Response(
                results_csv(rows),
                mimetype="text/csv",
                headers={
                    "Content-Disposition": (
                        "attachment; filename=classification-results.csv"
                    )
                },
            )
        except ValueError as exc:
            error = str(exc)
    return render_template("batch.html", error=error), 400 if error else 200


@pages.get("/sample.csv")
def sample_csv():
    """Provide synthetic text examples with no personal identifiers."""
    sample = (
        "Title,Text\n"
        "Dog chew toy,Durable rubber chew toy for my dog\n"
        "Vitamin supplement,Daily vitamin tablets for nutrition and wellness\n"
    )
    return Response(
        sample,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_batch.csv"},
    )


@pages.get("/categories")
def categories():
    """Browse taxonomy from the training partition only."""
    rows = classifier().taxonomy
    tops = sorted({row["Cat1"] for row in rows})
    selected = request.args.get("category", "")
    filtered = [row for row in rows if not selected or row["Cat1"] == selected]
    return render_template(
        "categories.html", rows=filtered, tops=tops, selected=selected
    )


@pages.get("/performance")
def performance():
    """Explain recorded results and the evaluation protocol."""
    return render_template(
        "performance.html",
        metrics=current_app.extensions["metrics"],
        metadata=classifier().metadata,
    )


@pages.get("/about")
def about():
    """Explain the problem, method and limitations in plain language."""
    return render_template("about.html")
