import urllib.request
import urllib
import os

# Safely creates parent/child folders. Does nothing if they already exist.
def download_pdf(pdf_url,pdf_title):

    if not os.path.exists("docsContainer"):
        os.mkdir("docsContainer")
    local_file_name = f'docsContainer/{pdf_title}.pdf'
    try:
        urllib.request.urlretrieve(pdf_url,local_file_name)
        print("Successfully save pdf")
    except Exception as e:
        print(f"Error Occured : {e}")

# for testing uncomment this 
# download_pdf("https://arxiv.org/pdf/2607.15273","MeanFlowNFT: Bringing Forward-Process RL to Average-Velocity Generators")

