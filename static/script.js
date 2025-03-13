document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('fileInput');
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to process the file');
        }

        const data = await response.json();
        document.getElementById('extractedText').textContent = data.extracted_text || 'No text extracted';
        document.getElementById('correctedText').textContent = data.corrected_text || 'No corrections made';
        document.getElementById('structuredNotes').textContent = data.structured_notes || 'No structured notes generated';
    } catch (error) {
        alert(error.message);
    }
});