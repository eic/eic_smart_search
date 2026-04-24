from app.ingestion.html import extract_html


def test_extract_html_removes_navigation_noise() -> None:
    html = """
    <html>
      <head><title>Example Page</title></head>
      <body>
        <nav>Home Documentation Policies</nav>
        <main>
          <h1>Run Tutorials</h1>
          <p>Use the documented simulation tutorial commands for local workflows.</p>
        </main>
        <footer>Repeated footer</footer>
      </body>
    </html>
    """

    extracted = extract_html(html, "https://example.test/tutorials")

    assert extracted.title == "Example Page"
    assert "Run Tutorials" in extracted.content
    assert "Repeated footer" not in extracted.content
    assert "Home Documentation Policies" not in extracted.content

