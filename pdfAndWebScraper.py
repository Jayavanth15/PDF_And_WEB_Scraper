import fitz  # PyMuPDF for PDF extraction
import faiss
from sentence_transformers import SentenceTransformer
import os
import openai
import json
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from selenium.webdriver import Remote, ChromeOptions
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SBR_WEBDRIVER = os.getenv("SBR_WEBDRIVER")

def extract_pdf_text(pdf_path):
    """Extracts text from a PDF and splits it into smaller chunks."""
    doc = fitz.open(pdf_path)
    return [page.get_text("text") for page in doc]

def extract_url_text(website):
    print("Launching chrome browser...")
    
    #chrome_driver_path = "C:\Users\jayavanth.k\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"
    sbr_connection = ChromiumRemoteConnection(SBR_WEBDRIVER, 'goog', 'chrome')
    with Remote(sbr_connection, options=ChromeOptions()) as driver:
        driver.get(website)
        solve_res = driver.execute(
            "executeCdpCommand",
            {
                "cmd" : "Captcha.waitForSolve",
                "params" : {"detectTimeout" : 10000},
            },
        )
        print('Captcha solve status : ', solve_res['value']['status'])
        print('Navigated! Scraping page content...')
        html = driver.page_source
        return html

def extract_body_content(html_content):

    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body
    if body_content:
        return str(body_content)
    return ""

def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, 'html.parser')

    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()
    
    cleaned_content = soup.get_text(separator="\n")
    cleaned_content = "\n".join(
        line.strip() for line in cleaned_content.splitlines() if line.strip()
    )
    return cleaned_content

def split_dom_content(dom_content, max_length = 6000):
    return [
        dom_content[i : i+max_length] for i in range(0, len(dom_content), max_length)
    ]

# def extract_url_text(url):
#     """Extracts text content from a given URL."""
#     try:
#         response = requests.get(url, timeout=10)
#         if response.status_code == 200:
#             soup = BeautifulSoup(response.text, 'html.parser')
#             return [p.text.strip() for p in soup.find_all('p') if p.text.strip()]
#         else:
#             print(f"Failed to retrieve {url}: Status code {response.status_code}")
#     except Exception as e:
#         print(f"Error fetching {url}: {e}")
#     return []

def create_faiss_index(text_chunks):
    """Creates a FAISS index from text embeddings."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(text_chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, text_chunks, model

def search_topics_faiss(index, text_chunks, model, topics, top_k=3):
    """Searches for the most relevant text across documents for given topics."""
    topic_embeddings = model.encode(topics, convert_to_numpy=True)
    _, indices = index.search(topic_embeddings, top_k)
    return {topic: [text_chunks[i] for i in idx_list] for topic, idx_list in zip(topics, indices)}

def response(topic, dom_chunks):
    """Processes extracted text using an LLM to generate summarized content."""
    prompt = f"""
        1. Precise the content in {dom_chunks}.
        2. Shorten the content exactly in 5-9 lines.
        3. Include any numerical data, percentages, or relevant values.
        4. Ensure clarity and coherence in the output.
        5. Give all negative(if present) as well as positive information combinely from given references
        6. Give only information ** no side headings **, ** no headings **.
        7. Don't give any junk values.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]
# Example Usage
pdf_paths = ["Mruti suzuki MSIL_IR_2023-24_HR.pdf"]
urls = ["https://www.icicidirect.com/research/equity/maruti-suzuki-(india)-ltd/14321",
        "https://www.ndtvprofit.com/markets/maruti-lists-major-risks-to-auto-sector-in-202324"
        ]
topics = ["Artificial Intelligence", "Inflation", "Taxation", "Regulatory Risks", "Foreign Exchange Risk", "Geo-political Tension", "Competitive Risk", "Critical Infrastructure Failure / Machine Breakdown", "Business Continuity / Sustainability", "Supply Chain Risk", "Commodity Price Risk", "Portfolio Risk", "Environmental Hazard Risk", "Workplace Accident", "Human Resource", "Financial Risk", "Breaches of Law (Local/International)", "Innovation Risk / Obsolete Technology", "Intellectual Property Risk", "Disruptive Technologies", "Data Compromise", "Cyber-crimes", "Counterfeiting", "Threat to Women Security", "Terrorism", "Natural Hazards", "Pandemic and Other Global Epidemic Diseases", "Resource Scarcity / Misutilisation / Overall Utilisation", "Public Sentiment", "Strategic Risk", "Delay in Execution of Projects", "Increased Number of Recalls and Quality Audits", "Failed / Hostile Mergers & Acquisitions"]

# Extract text from PDFs and URLs
text_chunks = []
for pdf in pdf_paths:
    text_chunks.extend(extract_pdf_text(pdf))
for url in urls[:5]:  # Limit to 5 URLs
    html_content = extract_url_text(url)
    body_content = extract_body_content(html_content)
    cleaned_content = clean_body_content(body_content)
    dom_chunks = split_dom_content(cleaned_content)
    text_chunks.extend(dom_chunks)
# Create FAISS index and search topics
index, text_chunks, model = create_faiss_index(text_chunks)
results = search_topics_faiss(index, text_chunks, model, topics)

# Process results with LLM
final_results = {}
for topic, matched_texts in results.items():
    ans = response(topic, matched_texts)
    if ans == f"There is no information related to {topic} in the provided text.":
        final_results[topic] = ""
    else :
        final_results[topic] = ans
    print(f"\n\n##########################################  {topic}  ###############################################\n\n")
    print(ans)

# Save results to JSON
with open("output.json", "w") as file:
    json.dump(final_results, file, indent=4)
