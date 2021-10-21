from flask import Flask, render_template, request, Response, flash 
from synth_module import VoiceSynthesis
 
app = Flask(__name__)
app.config["SECRET_KEY"] = "ABCD"
 
@app.route("/",  methods=['GET', 'POST'])
def index():
 
    return render_template('main.html'), 200

@app.route("/speech",  methods=['GET', 'POST'])
def speech():

    text = request.form['content']
    if not text:
        flash("텍스트를 입력해주세요")
        return render_template('main.html'), 200

    tts = VoiceSynthesis()
    path = tts.tts_flask(text)
    audio = path+".wav"

    return render_template('main.html', voice= audio, text = text), 200
 
@app.route("/<voice>", methods=['GET'])
def streamwav(voice):
    def generate():
        with open(voice, "rb") as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
    return Response(generate(), mimetype="audio/")

if __name__ == "__main__":
    app.run()
 
 
