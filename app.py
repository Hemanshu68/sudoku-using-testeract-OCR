from fileinput import filename
from pyexpat.errors import messages
from flask import Flask, render_template, request, flash, send_from_directory, redirect
import os
from werkzeug.utils import secure_filename
import cv2
from sudoku import Sudoku


UPLOAD_FOLDER = ""
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SESSION_TYPE"] = "memcached"
app.secret_key = "12344g"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["POST", "GET"])
def upload_file():
    if request.method == "POST":
        print(request.files["file"])
        if "file" not in request.files:
            ("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        print(file)
        if file and allowed_file(file.filename):
            filename = "input.png"
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        return redirect("result")
    return redirect("/")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/result", methods=["GET"])
def result():
    filename = "input.png"
    if os.path.exists(filename):
        s = Sudoku(filename)
        s.solve()
        return render_template(
            "results.html", original="input.png", solution="output.png"
        )
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
