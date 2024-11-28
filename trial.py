import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import PyPDF2
import docx
import os
import base64
from pathlib import Path

class PrivacyPolicySimplifier:
    def __init__(self):
        try:
            # Using T5-small model (much faster than BART-large)
            model_name = "t5-base"  # or "t5-base" for slightly better quality
            
            print("Loading model...")
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            
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
            # T5 requires a specific prefix for tasks
            input_text = "summarize: " + text
            
            inputs = self.tokenizer(input_text, 
                                  return_tensors="pt", 
                                  max_length=1024, 
                                  truncation=True)
            
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=300,
                min_length=100,
                num_beams=2,  # Reduced for speed
                length_penalty=1.5,
                early_stopping=True
            )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
    simplifier = PrivacyPolicySimplifier()
    
    # Add demo PDF files
    DEMO_POLICIES = {
        "Facebook Privacy Policy": "demo_policies/facebook_privacy.pdf",
        "Google Privacy Policy": "demo_policies/google_privacy.pdf",
        "Amazon Privacy Policy": "demo_policies/Amazon_Privacy.pdf",
        "Dun & Bradstreet Privacy Policy": "demo_policies/dun&bradstreet_privacy.pdf",
        "Identita Privacy Policy": "demo_policies/identita_privacy.pdf",
        "IndianOil Privacy Policy": "demo_policies/IndianOil_Privacy.pdf",
        "Flinders University Privacy Policy": "demo_policies/Flinders_Privacy.pdf",
        "Twitter Privacy Policy": "demo_policies/Twitter_Privacy.pdf",
        "Kpmg Privacy Policy": "demo_policies/Kpmg_Privacy.pdf",
        "Mark&Spencer Privacy Policy": "demo_policies/Mark&Spencer_privacy.pdf",
        "Apple Privacy Policy": "demo_policies/Apple_Privacy.pdf",
    }

    def process_demo_policy(demo_choice):
        if not demo_choice:
            return "Please select a demo policy"
        
        file_path = DEMO_POLICIES[demo_choice]
        try:
            with open(file_path, 'rb') as file:
                text = simplifier.extract_text_from_file(file)
                if text:
                    return simplifier.simplify(text)
                return "Error processing the demo file"
        except Exception as e:
            return f"Error: {str(e)}"

    def process_input(text_input, file_input, demo_choice):
        if demo_choice:
            return process_demo_policy(demo_choice)
        elif text_input:
            return simplifier.simplify(text_input)
        elif file_input:
            text = simplifier.extract_text_from_file(file_input)
            if text is None:
                return "Error: Could not process file. Please ensure it's a PDF or DOCX file."
            return simplifier.simplify(text)
        else:
            return "Please provide input using one of the available methods."

    # Create the Gradio interface
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📄 Privacy Policy Simplifier
            ### Make complex privacy policies easy to understand! 🎯
            
            Try our tool with one of these options:
            1. Use a demo privacy policy from popular companies
            2. Paste your own privacy policy text
            3. Upload a PDF/DOCX file
            """
        )
        
        with gr.Column():
            # Demo policies dropdown
            gr.Markdown("### Option 1: Try with Demo Privacy Policies")
            demo_policy = gr.Dropdown(
                choices=list(DEMO_POLICIES.keys()),
                label="Select a company's privacy policy",
                value=None
            )
            
            with gr.Accordion("📌 About Demo Policies", open=False):
                gr.Markdown("""
                    These are real privacy policies from popular companies:
                    - Facebook's privacy policy demonstrates social media data handling
                    - Google's policy shows comprehensive data collection practices
                    - Amazon's policy illustrates e-commerce data usage
                    
                    Select any policy to see how our tool simplifies it!
                """)
            
            gr.Markdown("### OR")
            
            # Text input
            text_input = gr.Textbox(
                label="Option 2: Paste Privacy Policy Text ✍️",
                placeholder="Paste your privacy policy text here...",
                lines=10
            )
            
            gr.Markdown("### OR")
            
            # File upload
            file_input = gr.File(
                label="Option 3: Upload Privacy Policy File 📎",
                file_types=[".pdf", ".docx"],
                type="filepath"
            )
            
            # Output text
            output_text = gr.Textbox(
                label="Simplified Explanation 🔍",
                lines=15
            )
            
            # Simplify button
            simplify_button = gr.Button(
                "Simplify ✨", 
                variant="primary",
                size="lg"
            )
            
            gr.Markdown(
                """
                ### 🎯 How to Use:
                1. **Try Demo Policies**: Select from our pre-loaded company policies
                2. **Custom Input**: Paste your own text or upload a file
                3. **Get Results**: Click 'Simplify' to see the breakdown
                
                ### 💡 Pro Tips:
                - Start with demo policies to see how the tool works
                - For best results, provide complete privacy policy sections
                - Supported formats: PDF, DOCX
                """
            )
        
        simplify_button.click(
            fn=process_input,
            inputs=[text_input, file_input, demo_policy],
            outputs=output_text
        )
        
    return demo

# Create and launch the demo
demo = create_gradio_interface()

if __name__ == "__main__":
    demo.launch()
else:
    # For HF Spaces
    demo.queue(max_size=5)

