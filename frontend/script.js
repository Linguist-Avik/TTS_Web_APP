async function generateSpeech() {
    const text = document.getElementById("textInput").value;
    const language = document.getElementById("language").value;

    const response = await fetch("http://127.0.0.1:8000/synthesize/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, language }),
    });

    const data = await response.json();
    if (data.file) {
        document.getElementById("audioPlayer").src = data.file;
    } else {
        alert("Error generating speech");
    }
}
