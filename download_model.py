import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def download_model():
    # Create directory
    os.makedirs("./models/legal-bert", exist_ok=True)
    
    # Model name
    model_name = "nlegaldomain/legalbert-summarization-privacy-policy"
    
    print("Downloading model files...")
    
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token="Yhf_rZgDeDHfwfsyVxnaBpIpdaZURKPDTlgisX"  # Replace with your token
        )
        
        # Save tokenizer files
        tokenizer.save_pretrained("./models/legal-bert")
        print("Tokenizer saved successfully!")
        
        # Download model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            use_auth_token="Yhf_rZgDeDHfwfsyVxnaBpIpdaZURKPDTlgisX"  # Replace with your token
        )
        
        # Save model files
        model.save_pretrained("./models/legal-bert")
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    download_model() 