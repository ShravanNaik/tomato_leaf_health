import sys
import importlib
importlib.import_module('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import openai
from PIL import Image
import base64
import io
import json
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="Advanced Tomato Disease Detection",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_footer_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* This targets GitHub icon in the footer */
    .st-emotion-cache-1y4p8pa.ea3mdgi1 {
        display: none !important;
    }

    /* This targets the entire footer area */
    .st-emotion-cache-164nlkn {
        display: none !important;
    }
    </style>
"""

st.markdown(hide_footer_style, unsafe_allow_html=True)

class TomatoAnalysisAgent:
    """Multi-agent system for comprehensive plant disease analysis"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        
    def encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def pathology_agent(self, image: Image.Image) -> Dict[str, Any]:
        """Plant Pathology Specialist Agent"""
        base64_image = self.encode_image(image)
        
        prompt = """
        You are Dr. Sarah Chen, a world-renowned plant pathologist with 20 years of experience in tomato diseases.
        
        Analyze this tomato leaf image for diseases. Focus on:
        
        FUNGAL DISEASES:
        - Early Blight (Alternaria solani) - brown spots with concentric rings
        - Late Blight (Phytophthora infestans) - water-soaked lesions
        - Septoria Leaf Spot - small circular spots with gray centers
        - Target Spot (Corynespora cassiicola) - circular lesions with target pattern
        - Anthracnose - sunken lesions on mature fruit
        - Powdery Mildew - white powdery coating
        - Downy Mildew - yellow patches with fuzzy growth
        - Fusarium Wilt - yellowing and wilting from bottom up
        - Verticillium Wilt - V-shaped yellowing
        - Black Mold (Alternaria alternata) - dark lesions
        - Gray Mold (Botrytis cinerea) - gray fuzzy growth
        - Leaf Mold (Passalora fulva) - olive-green patches
        
        BACTERIAL DISEASES:
        - Bacterial Spot (Xanthomonas) - small dark spots with yellow halos
        - Bacterial Speck (Pseudomonas syringae) - tiny black spots
        - Bacterial Wilt (Ralstonia solanacearum) - sudden wilting
        - Bacterial Canker (Clavibacter michiganensis) - cankers on stems
        - Pith Necrosis - hollow brown pith in stems
        
        VIRAL DISEASES:
        - Tomato Mosaic Virus (ToMV) - mottled yellow-green patterns
        - Tobacco Mosaic Virus (TMV) - mosaic patterns
        - Tomato Spotted Wilt Virus - bronze spots and rings
        - Cucumber Mosaic Virus - stunted growth, mottling
        - Tomato Yellow Leaf Curl Virus - upward curling leaves
        - Tomato Bushy Stunt Virus - stunted bushy growth
        
        Provide analysis in JSON format:
        {
            "agent_name": "Plant Pathology Specialist",
            "diseases_identified": ["list of diseases with confidence %"],
            "pathogen_type": "fungal/bacterial/viral/physiological",
            "disease_stage": "early/intermediate/advanced",
            "severity_score": "1-10 scale",
            "key_symptoms": ["detailed symptom list"],
            "differential_diagnosis": ["possible alternative diseases"],
            "prognosis": "likely outcome if untreated"
        }
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a plant pathology expert. Always respond with valid JSON."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1200
            )
            
            content = response.choices[0].message.content
            return self._parse_json_response(content)
            
        except Exception as e:
            return {"error": str(e), "agent_name": "Plant Pathology Specialist"}
    
    def entomology_agent(self, image: Image.Image) -> Dict[str, Any]:
        """Entomology Specialist Agent for pest damage"""
        base64_image = self.encode_image(image)
        
        prompt = """
        You are Dr. Marcus Rodriguez, an entomologist specializing in tomato pests and their damage patterns.
        
        Analyze this image for pest damage signs:
        
        INSECT PESTS:
        - Hornworms - large holes, black droppings
        - Cutworms - stems cut at soil level
        - Aphids - yellowing, sticky honeydew, curled leaves
        - Whiteflies - yellowing, stunted growth
        - Thrips - silver streaks, black specks
        - Spider Mites - stippling, webbing, bronze appearance
        - Flea Beetles - small round holes
        - Colorado Potato Beetles - large irregular holes
        - Leaf Miners - serpentine tunnels in leaves
        - Stink Bugs - cloudy spot on fruit, feeding damage
        - Psyllids - yellowing, twisted growth
        - Scale Insects - yellow spots, honeydew
        
        MITE DAMAGE:
        - Two-spotted Spider Mites - stippling, webbing
        - Broad Mites - distorted growth, bronzing
        - Cyclamen Mites - stunted, distorted leaves
        
        OTHER ARTHROPODS:
        - Slugs/Snails - irregular holes, slime trails
        - Nematodes - root galls, stunted growth
        
        Provide JSON analysis:
        {
            "agent_name": "Entomology Specialist",
            "pest_damage_detected": ["list of pest damage with confidence %"],
            "damage_pattern": "description of feeding damage",
            "pest_lifecycle_stage": "egg/larva/adult damage",
            "infestation_level": "light/moderate/heavy",
            "secondary_issues": ["diseases that follow pest damage"],
            "beneficial_insects": ["predators that might help"]
        }
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an entomology expert. Always respond with valid JSON."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            return self._parse_json_response(content)
            
        except Exception as e:
            return {"error": str(e), "agent_name": "Entomology Specialist"}
    
    def nutrition_agent(self, image: Image.Image) -> Dict[str, Any]:
        """Plant Nutrition Specialist Agent"""
        base64_image = self.encode_image(image)
        
        prompt = """
        You are Dr. Lisa Thompson, a plant nutrition expert specializing in tomato nutrient disorders.
        
        Analyze this image for nutritional deficiencies and disorders:
        
        NUTRIENT DEFICIENCIES:
        - Nitrogen (N) - yellowing of older leaves, stunted growth
        - Phosphorus (P) - purple/reddish leaves, poor fruit development
        - Potassium (K) - leaf edge burn, poor fruit quality
        - Calcium (Ca) - blossom end rot, tip burn
        - Magnesium (Mg) - interveinal yellowing of older leaves
        - Iron (Fe) - interveinal yellowing of young leaves
        - Manganese (Mn) - interveinal yellowing, brown spots
        - Zinc (Zn) - small leaves, shortened internodes
        - Boron (B) - brittle leaves, poor fruit set
        - Copper (Cu) - wilting, blue-green leaves
        - Sulfur (S) - yellowing of young leaves
        - Molybdenum (Mo) - yellowing, cupping of leaves
        
        PHYSIOLOGICAL DISORDERS:
        - Blossom End Rot - calcium deficiency/water stress
        - Catfacing - temperature/nutrition issues
        - Cracking - water fluctuations
        - Sunscald - excessive heat/light exposure
        - Edema - overwatering, poor drainage
        - Puffiness - cool temperatures, poor pollination
        
        Provide JSON analysis:
        {
            "agent_name": "Plant Nutrition Specialist",
            "nutrient_deficiencies": ["deficiencies with severity %"],
            "physiological_disorders": ["disorders identified"],
            "soil_ph_indication": "acidic/neutral/alkaline suggestion",
            "fertilizer_recommendations": ["specific nutrient needs"],
            "environmental_factors": ["contributing conditions"]
        }
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a plant nutrition expert. Always respond with valid JSON."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            return self._parse_json_response(content)
            
        except Exception as e:
            return {"error": str(e), "agent_name": "Plant Nutrition Specialist"}
    
    def environmental_agent(self, image: Image.Image) -> Dict[str, Any]:
        """Environmental Stress Specialist Agent"""
        base64_image = self.encode_image(image)
        
        prompt = """
        You are Dr. Ahmed Hassan, an environmental plant stress specialist.
        
        Analyze this image for environmental stress factors:
        
        ABIOTIC STRESS:
        - Heat Stress - leaf curling, wilting, sunscald
        - Cold Stress - purple/blue coloration, stunted growth
        - Water Stress - wilting, leaf drop, blossom end rot
        - Light Stress - etiolation, sunscald, poor color
        - Wind Damage - torn leaves, broken stems
        - Hail Damage - puncture wounds, bruising
        - Chemical Burn - leaf margins, spotting
        - Salt Stress - leaf burn, stunted growth
        - Oxygen Stress - yellowing, root problems
        - Transplant Shock - wilting, yellowing
        
        ENVIRONMENTAL CONDITIONS:
        - Humidity Issues - fungal problems, poor pollination
        - Air Circulation - disease pressure, poor growth
        - Soil Compaction - stunted roots, yellowing
        - pH Problems - nutrient lockout, poor growth
        - Contamination - unusual symptoms, poor health
        
        Provide JSON analysis:
        {
            "agent_name": "Environmental Stress Specialist",
            "stress_factors": ["environmental stresses with severity"],
            "climate_conditions": ["likely growing conditions"],
            "soil_conditions": ["soil health indicators"],
            "water_management": ["irrigation recommendations"],
            "microclimate_factors": ["local environment issues"]
        }
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an environmental stress expert. Always respond with valid JSON."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            return self._parse_json_response(content)
            
        except Exception as e:
            return {"error": str(e), "agent_name": "Environmental Stress Specialist"}
    
    def treatment_agent(self, pathology_data: Dict, entomology_data: Dict, 
                       nutrition_data: Dict, environmental_data: Dict) -> Dict[str, Any]:
        """Treatment Coordinator Agent"""
        
        prompt = f"""
        You are Dr. Jennifer Park, an integrated pest management specialist and treatment coordinator.
        
        Based on the following multi-agent analysis results, provide comprehensive treatment recommendations:
        
        PATHOLOGY FINDINGS: {json.dumps(pathology_data, indent=2)}
        ENTOMOLOGY FINDINGS: {json.dumps(entomology_data, indent=2)}
        NUTRITION FINDINGS: {json.dumps(nutrition_data, indent=2)}
        ENVIRONMENTAL FINDINGS: {json.dumps(environmental_data, indent=2)}
        
        Provide integrated treatment plan in JSON format:
        {{
            "agent_name": "Treatment Coordinator",
            "priority_treatments": ["immediate actions needed"],
            "organic_treatments": ["natural/organic solutions"],
            "chemical_treatments": ["conventional options if needed"],
            "cultural_practices": ["growing practice changes"],
            "prevention_strategies": ["long-term prevention"],
            "monitoring_schedule": ["what to watch for"],
            "treatment_timeline": ["when to apply treatments"],
            "resistance_management": ["avoiding resistance issues"],
            "integrated_approach": ["holistic management strategy"]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an integrated treatment specialist. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200
            )
            
            content = response.choices[0].message.content
            return self._parse_json_response(content)
            
        except Exception as e:
            return {"error": str(e), "agent_name": "Treatment Coordinator"}
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from response content"""
        try:
            # Try to find JSON in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # If no JSON found, create a fallback response
                return {
                    "raw_response": content,
                    "parsing_error": "No valid JSON found in response",
                    "agent_name": "Unknown"
                }
        except json.JSONDecodeError as e:
            return {
                "raw_response": content,
                "parsing_error": f"JSON decode error: {str(e)}",
                "agent_name": "Unknown"
            }
    
    def run_multi_agent_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Run all agents sequentially for comprehensive analysis"""
        
        results = {}
        
        try:
            # Run pathology agent
            results["pathology"] = self.pathology_agent(image)
            
            # Run entomology agent
            results["entomology"] = self.entomology_agent(image)
            
            # Run nutrition agent
            results["nutrition"] = self.nutrition_agent(image)
            
            # Run environmental agent
            results["environmental"] = self.environmental_agent(image)
            
            # Run treatment coordinator with all results
            results["treatment"] = self.treatment_agent(
                results["pathology"], results["entomology"], 
                results["nutrition"], results["environmental"]
            )
            
            results["analysis_timestamp"] = datetime.now().isoformat()
            return results
            
        except Exception as e:
            return {"error": f"Multi-agent analysis failed: {str(e)}"}

def display_agent_results(results: Dict[str, Any]):
    """Display results from all agents in organized tabs"""
    
    if not results or "analysis_timestamp" not in results:
        st.error("No valid analysis results to display")
        return
    
    st.success(f"✅ Multi-Agent Analysis Complete - {results['analysis_timestamp']}")
    
    # Create tabs for each agent
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🦠 Pathology", "🐛 Entomology", "🌱 Nutrition", 
        "🌤️ Environment", "💊 Treatment", "📊 Summary"
    ])
    
    with tab1:
        st.header("🦠 Plant Pathology Analysis")
        pathology = results.get("pathology", {})
        if "error" not in pathology:
            
            if "diseases_identified" in pathology:
                st.subheader("Diseases Identified")
                for disease in pathology["diseases_identified"]:
                    st.error(f"🔴 {disease}")
            
            col1, col2 = st.columns(2)
            with col1:
                if "pathogen_type" in pathology:
                    st.metric("Pathogen Type", pathology["pathogen_type"])
                if "severity_score" in pathology:
                    st.metric("Severity Score", f"{pathology['severity_score']}/10")
            
            with col2:
                if "disease_stage" in pathology:
                    st.metric("Disease Stage", pathology["disease_stage"])
            
            if "key_symptoms" in pathology:
                st.subheader("Key Symptoms")
                for symptom in pathology["key_symptoms"]:
                    st.write(f"• {symptom}")
            
            if "differential_diagnosis" in pathology:
                st.subheader("Differential Diagnosis")
                for diagnosis in pathology["differential_diagnosis"]:
                    st.info(f"📋 {diagnosis}")
        else:
            st.error(f"Pathology analysis error: {pathology.get('error', 'Unknown error')}")
    
    with tab2:
        st.header("🐛 Entomology Analysis")
        entomology = results.get("entomology", {})
        if "error" not in entomology:
            
            if "pest_damage_detected" in entomology:
                st.subheader("Pest Damage Detected")
                for damage in entomology["pest_damage_detected"]:
                    st.warning(f"🟡 {damage}")
            
            if "infestation_level" in entomology:
                level = entomology["infestation_level"]
                if level.lower() == "light":
                    st.success(f"Infestation Level: {level}")
                elif level.lower() == "moderate":
                    st.warning(f"Infestation Level: {level}")
                else:
                    st.error(f"Infestation Level: {level}")
            
            if "damage_pattern" in entomology:
                st.subheader("Damage Pattern")
                st.write(entomology["damage_pattern"])
            
            if "beneficial_insects" in entomology:
                st.subheader("Beneficial Insects")
                for insect in entomology["beneficial_insects"]:
                    st.success(f"🟢 {insect}")
        else:
            st.error(f"Entomology analysis error: {entomology.get('error', 'Unknown error')}")
    
    with tab3:
        st.header("🌱 Plant Nutrition Analysis")
        nutrition = results.get("nutrition", {})
        if "error" not in nutrition:
            
            if "nutrient_deficiencies" in nutrition:
                st.subheader("Nutrient Deficiencies")
                for deficiency in nutrition["nutrient_deficiencies"]:
                    st.error(f"🔴 {deficiency}")
            
            if "physiological_disorders" in nutrition:
                st.subheader("Physiological Disorders")
                for disorder in nutrition["physiological_disorders"]:
                    st.warning(f"🟡 {disorder}")
            
            col1, col2 = st.columns(2)
            with col1:
                if "soil_ph_indication" in nutrition:
                    st.metric("Soil pH Indication", nutrition["soil_ph_indication"])
            
            if "fertilizer_recommendations" in nutrition:
                st.subheader("Fertilizer Recommendations")
                for rec in nutrition["fertilizer_recommendations"]:
                    st.info(f"💡 {rec}")
        else:
            st.error(f"Nutrition analysis error: {nutrition.get('error', 'Unknown error')}")
    
    with tab4:
        st.header("🌤️ Environmental Stress Analysis")
        environmental = results.get("environmental", {})
        if "error" not in environmental:
            
            if "stress_factors" in environmental:
                st.subheader("Environmental Stress Factors")
                for stress in environmental["stress_factors"]:
                    st.warning(f"⚠️ {stress}")
            
            if "climate_conditions" in environmental:
                st.subheader("Climate Conditions")
                for condition in environmental["climate_conditions"]:
                    st.info(f"🌡️ {condition}")
            
            if "water_management" in environmental:
                st.subheader("Water Management")
                for rec in environmental["water_management"]:
                    st.success(f"💧 {rec}")
        else:
            st.error(f"Environmental analysis error: {environmental.get('error', 'Unknown error')}")
    
    with tab5:
        st.header("💊 Integrated Treatment Plan")
        treatment = results.get("treatment", {})
        if "error" not in treatment:
            
            if "priority_treatments" in treatment:
                st.subheader("🚨 Priority Treatments (Immediate Action)")
                for priority in treatment["priority_treatments"]:
                    st.error(f"🔴 {priority}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if "organic_treatments" in treatment:
                    st.subheader("🌿 Organic Treatments")
                    for organic in treatment["organic_treatments"]:
                        st.success(f"🟢 {organic}")
                
                if "cultural_practices" in treatment:
                    st.subheader("🌾 Cultural Practices")
                    for practice in treatment["cultural_practices"]:
                        st.info(f"📋 {practice}")
            
            with col2:
                if "chemical_treatments" in treatment:
                    st.subheader("⚗️ Chemical Treatments")
                    for chemical in treatment["chemical_treatments"]:
                        st.warning(f"🟡 {chemical}")
                
                if "prevention_strategies" in treatment:
                    st.subheader("🛡️ Prevention Strategies")
                    for prevention in treatment["prevention_strategies"]:
                        st.info(f"🔵 {prevention}")
            
            if "treatment_timeline" in treatment:
                st.subheader("📅 Treatment Timeline")
                for timeline in treatment["treatment_timeline"]:
                    st.write(f"⏰ {timeline}")
        else:
            st.error(f"Treatment analysis error: {treatment.get('error', 'Unknown error')}")
    
    with tab6:
        st.header("📊 Analysis Summary")
        
        # Create summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pathology = results.get("pathology", {})
            disease_count = len(pathology.get("diseases_identified", []))
            st.metric("Diseases Found", disease_count)
        
        with col2:
            entomology = results.get("entomology", {})
            pest_count = len(entomology.get("pest_damage_detected", []))
            st.metric("Pest Issues", pest_count)
        
        with col3:
            nutrition = results.get("nutrition", {})
            deficiency_count = len(nutrition.get("nutrient_deficiencies", []))
            st.metric("Nutrient Issues", deficiency_count)
        
        with col4:
            environmental = results.get("environmental", {})
            stress_count = len(environmental.get("stress_factors", []))
            st.metric("Stress Factors", stress_count)
        
        # Overall health assessment
        total_issues = disease_count + pest_count + deficiency_count + stress_count
        
        st.subheader("Overall Plant Health Assessment")
        if total_issues == 0:
            st.success("🌱 Plant appears healthy with no major issues detected")
        elif total_issues <= 3:
            st.warning(f"⚠️ Plant has {total_issues} issues that need attention")
        else:
            st.error(f"🚨 Plant has {total_issues} serious issues requiring immediate intervention")
        
        # Generate downloadable report
        if st.button("📄 Generate Detailed Report"):
            report_data = {
                "Analysis Date": results["analysis_timestamp"],
                "Total Issues Found": total_issues,
                "Diseases": disease_count,
                "Pest Problems": pest_count,
                "Nutrition Issues": deficiency_count,
                "Environmental Stress": stress_count,
                "Detailed Results": results
            }
            
            st.download_button(
                label="Download Full Analysis Report (JSON)",
                data=json.dumps(report_data, indent=2),
                file_name=f"tomato_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def main():
    st.title("🍅 Tomato Disease Detection System")
    st.markdown("**Powered by OpenAI and Specialized AI Agents**")
    
    st.info("""
    🤖 **Five Specialized AI Agents Working Together:**
    - **Plant Pathology Specialist**: Detects 30+ diseases (fungal, bacterial, viral)
    - **Entomology Specialist**: Identifies pest damage and infestations
    - **Plant Nutrition Specialist**: Diagnoses nutrient deficiencies and disorders
    - **Environmental Stress Specialist**: Analyzes abiotic stress factors
    - **Treatment Coordinator**: Provides integrated management strategies
    """)
    
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("❌ OpenAI API key not found!")
        st.warning("""
        **Please set up your OpenAI API key:**
        
        1. Create a `.env` file in the same directory as this app
        2. Add the following line to the `.env` file:
        ```
        OPENAI_API_KEY=your_api_key_here
        ```
        3. Replace `your_api_key_here` with your actual OpenAI API key
        4. Restart the Streamlit app
        
        **How to get OpenAI API Key:**
        - Visit https://platform.openai.com/
        - Create account and navigate to API Keys
        - Generate new API key
        - Copy and paste it in the .env file
        """)
        return
    
    # Display API key status (masked for security)
    st.sidebar.success(f"🔑 API Key Loaded: {'*' * 20}{api_key[-4:]}")
    
    # Initialize agent manager
    agent_manager = TomatoAnalysisAgent(api_key)
    
    # File upload section
    st.header("📤 Upload Tomato Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose a tomato leaf image for comprehensive analysis...",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Upload a clear, well-lit image of the tomato leaf showing any symptoms"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image details
            st.subheader("Image Information")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Format:** {image.format}")
            st.write(f"**File Size:** {uploaded_file.size / 1024:.1f} KB")
        
        with col2:
            st.subheader("🚀 Multi-Agent Analysis")
            st.write("Click below to start comprehensive analysis using 5 specialized AI agents")
            
            if st.button("🔍 Start Multi-Agent Analysis", type="primary", use_container_width=True):
                
                with st.spinner("🤖 Running multi-agent analysis..."):
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Run pathology agent
                        status_text.text("🦠 Plant Pathology Agent analyzing diseases...")
                        progress_bar.progress(20)
                        
                        # Run the multi-agent analysis
                        results = agent_manager.run_multi_agent_analysis(image)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")
                        
                        # Store results in session state
                        st.session_state['analysis_results'] = results
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.write("Please check your API key and try again.")
                        return
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
        
        # Display results if available
        if 'analysis_results' in st.session_state:
            st.markdown("---")
            display_agent_results(st.session_state['analysis_results'])
    
    # Sidebar information (removed configuration section)
    st.sidebar.header("🧠 AI Agent Capabilities")
    
    with st.sidebar.expander("🦠 Plant Pathology Agent"):
        st.markdown("""
        **Detects 30+ diseases:**
        - **Fungal**: Early Blight, Late Blight, Septoria, Target Spot, Anthracnose, Powdery Mildew, Fusarium Wilt, etc.
        - **Bacterial**: Bacterial Spot, Bacterial Speck, Bacterial Wilt, Bacterial Canker
        - **Viral**: TMV, ToMV, TSWV, CMV, TYLCV, TBSV
        """)
    
    with st.sidebar.expander("🐛 Entomology Agent"):
        st.markdown("""
        **Identifies pest damage:**
        - **Insects**: Hornworms, Aphids, Whiteflies, Thrips, Spider Mites, Flea Beetles
        - **Damage Patterns**: Feeding holes, stippling, tunnels, distortion
        - **Beneficial Insects**: Natural predators and biological control
        """)
    
    with st.sidebar.expander("🌱 Nutrition Agent"):
        st.markdown("""
        **Diagnoses deficiencies:**
        - **Macronutrients**: N, P, K, Ca, Mg, S
        - **Micronutrients**: Fe, Mn, Zn, B, Cu, Mo
        - **Disorders**: Blossom end rot, catfacing, cracking, sunscald
        """)
    
    with st.sidebar.expander("🌤️ Environmental Agent"):
        st.markdown("""
        **Analyzes stress factors:**
        - **Abiotic Stress**: Heat, cold, water, light, wind, chemicals
        - **Growing Conditions**: Humidity, air circulation, soil health
        - **Management**: Irrigation, climate control recommendations
        """)
    
    with st.sidebar.expander("💊 Treatment Coordinator"):
        st.markdown("""
        **Integrated management:**
        - **Treatment Priority**: Immediate vs. long-term actions
        - **Organic Solutions**: Natural and biological controls
        - **Chemical Options**: Conventional treatments when needed
        - **Prevention**: Cultural practices and resistance management
        """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("📸 Image Guidelines")
    st.sidebar.markdown("""
    **For best results:**
    - 🔆 Good lighting (natural light preferred)
    - 🎯 Sharp focus on symptoms
    - 📏 Close-up of affected areas
    - 🍃 Include both healthy and affected tissue
    - 📱 High resolution (>1MP recommended)
    
    **Avoid:**
    - ❌ Blurry or dark images
    - ❌ Images with heavy shadows
    - ❌ Too far from subject
    - ❌ Low resolution photos
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔬 Disease Database")
    
    # Expandable disease reference
    with st.sidebar.expander("📚 Common Tomato Diseases"):
        disease_data = {
            "Fungal Diseases": [
                "Early Blight (Alternaria solani)",
                "Late Blight (Phytophthora infestans)", 
                "Septoria Leaf Spot",
                "Target Spot (Corynespora cassiicola)",
                "Anthracnose",
                "Powdery Mildew",
                "Downy Mildew",
                "Fusarium Wilt",
                "Verticillium Wilt",
                "Black Mold",
                "Gray Mold (Botrytis)",
                "Leaf Mold (Passalora fulva)"
            ],
            "Bacterial Diseases": [
                "Bacterial Spot (Xanthomonas)",
                "Bacterial Speck (Pseudomonas)",
                "Bacterial Wilt (Ralstonia)",
                "Bacterial Canker (Clavibacter)",
                "Pith Necrosis"
            ],
            "Viral Diseases": [
                "Tomato Mosaic Virus (ToMV)",
                "Tobacco Mosaic Virus (TMV)",
                "Tomato Spotted Wilt Virus",
                "Cucumber Mosaic Virus",
                "Tomato Yellow Leaf Curl Virus",
                "Tomato Bushy Stunt Virus"
            ]
        }
        
        for category, diseases in disease_data.items():
            st.markdown(f"**{category}:**")
            for disease in diseases:
                st.markdown(f"• {disease}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>🍅 Tomato Disease Detection System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
