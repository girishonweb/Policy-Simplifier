import gradio as gr
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import PyPDF2
import docx

class PrivacyPolicySimplifier:
    def __init__(self):
        try:
            # Using BART for better simplification
            model_name = "facebook/bart-large-cnn"
            
            print("Loading model...")
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise
    
    def extract_text_from_file(self, file):
        """Extract text from uploaded PDF or DOCX file"""
        try:
            if file.name.endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
            elif file.name.endswith('.docx'):
                doc = docx.Document(file)
                text = ""
                for para in doc.paragraphs:
                    text += para.text + "\n"
                return text
            else:
                return None
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return None

    def simplify(self, text):
        try:
            # Add prompt for simplified explanation in bullet points
            input_text = "Explain this privacy policy in simple terms and organize key points: " + text
            
            inputs = self.tokenizer(input_text, 
                                  return_tensors="pt", 
                                  max_length=1024, 
                                  truncation=True)
            
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=300,  # Increased for more detailed explanation
                min_length=100,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Process the summary into organized bullet points
            return self.format_into_sections(summary)
            
        except Exception as e:
            return f"Error during simplification: {str(e)}"

    def format_into_sections(self, text):
        """Format the summary into organized sections with bullet points"""
        sections = [
            "Data Collection:",
            "Data Usage:",
            "Data Sharing:",
            "Your Rights:",
            "Security Measures:"
        ]
        
        formatted_text = "Privacy Policy Simplified Explanation:\n\n"
        points = text.split('. ')
        
        for section in sections:
            # Make section headings bold
            formatted_text += f"**{section}**\n"
            relevant_points = [p for p in points if any(keyword in p.lower() 
                             for keyword in section.lower().split())]
            if relevant_points:
                for point in relevant_points:
                    if point.strip():
                        formatted_text += f"• {point.strip()}\n"
            else:
                formatted_text += "• No specific information provided.\n"
            formatted_text += "\n"
        
        return formatted_text

def create_gradio_interface():
    # Initialize the simplifier
    simplifier = PrivacyPolicySimplifier()
    
    # Define the processing functions
    def process_input(text_input, file_input):
        if text_input and file_input is not None:
            return "Please use either text input OR file upload, not both."
        
        if text_input:
            return simplifier.simplify(text_input)
        elif file_input is not None:
            text = simplifier.extract_text_from_file(file_input)
            if text is None:
                return "Error: Could not process file. Please ensure it's a PDF or DOCX file."
            return simplifier.simplify(text)
        else:
            return "Please provide either text input or upload a file."

    # Create the Gradio interface with improved UI
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📄 Privacy Policy Simplifier
            ### Make complex privacy policies easy to understand! 🎯
            """
        )
        
        with gr.Column():
            gr.Markdown("### Choose your input method 👇")
            
            # Text input
            text_input = gr.Textbox(
                label="Option 1: Paste Privacy Policy Text ✍️",
                placeholder="Paste your privacy policy text here...",
                lines=10
            )
            
            gr.Markdown("### OR")
            
            # File upload
            file_input = gr.File(
                label="Option 2: Upload Privacy Policy File 📎",
                file_types=[".pdf", ".docx"],
                type="filepath"
            )
            
            # Output text
            output_text = gr.Textbox(
                label="Simplified Explanation 🔍",
                lines=15
            )
            
            # Simplify button with better styling
            simplify_button = gr.Button(
                "Simplify ✨", 
                variant="primary",
                size="lg"
            )
            
            # Add some helpful information
            gr.Markdown(
                """
                ### ℹ️ Instructions:
                1. Choose **ONE** input method:
                   - Either paste your text directly
                   - Or upload a PDF/DOCX file
                2. Click the 'Simplify' button
                3. Get your simplified explanation!
                
                ### 📌 Note:
                - Supported file formats: PDF, DOCX
                - For best results, ensure clear and complete privacy policy text
                """
            )
        
        simplify_button.click(
            fn=process_input,
            inputs=[text_input, file_input],
            outputs=output_text
        )
        
    return demo

# Main execution
if __name__ == "__main__":
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

