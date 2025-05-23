"""
Health reminder and tracking tool for elderly users
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import re

from src.tools.base import BaseTool, ToolResult
from src.memory.schemas import HealthInfo, DailyRoutine


class HealthReminderTool(BaseTool):
    """Tool for health reminders and medication tracking"""
    
    @property
    def name(self) -> str:
        return "health_reminder"
    
    @property
    def description(self) -> str:
        return "Check medication schedules, set health reminders, and provide health tips"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Health-related query or reminder request"
                },
                "context": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["check_medication", "set_reminder", "health_tip", "symptom_check"],
                            "description": "Type of health action"
                        },
                        "user_health_info": {
                            "type": "object",
                            "description": "User's health information from memory"
                        },
                        "time": {
                            "type": "string",
                            "description": "Time for reminder (if applicable)"
                        }
                    }
                }
            },
            "required": ["query"]
        }
    
    async def _execute_impl(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Execute health-related queries"""
        
        action = context.get("action", "general") if context else "general"
        user_health = context.get("user_health_info", {}) if context else {}
        
        # Determine action from query if not specified
        if action == "general":
            action = self._determine_action(query)
        
        try:
            if action == "check_medication":
                result = self._check_medications(user_health)
            elif action == "set_reminder":
                result = self._set_health_reminder(query, context)
            elif action == "health_tip":
                result = self._provide_health_tip(query, user_health)
            elif action == "symptom_check":
                result = self._basic_symptom_check(query)
            else:
                result = self._general_health_response(query)
            
            return ToolResult(
                success=True,
                data=result,
                metadata={
                    "action": action,
                    "health_related": True
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Health tool error: {str(e)}"
            )
    
    def _determine_action(self, query: str) -> str:
        """Determine the health action from query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["medication", "medicine", "pill", "dose"]):
            return "check_medication"
        elif any(word in query_lower for word in ["remind", "reminder", "alert"]):
            return "set_reminder"
        elif any(word in query_lower for word in ["tip", "advice", "healthy", "wellness"]):
            return "health_tip"
        elif any(word in query_lower for word in ["symptom", "pain", "feel", "hurt"]):
            return "symptom_check"
        else:
            return "general"
    
    def _check_medications(self, user_health: Dict[str, Any]) -> Dict[str, Any]:
        """Check medication schedule"""
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Mock medication schedule (in production, read from user's health info)
        medications = user_health.get("medications", [
            {
                "name": "Blood Pressure Medicine",
                "schedule": ["08:00", "20:00"],
                "instructions": "Take with food"
            },
            {
                "name": "Vitamin D",
                "schedule": ["09:00"],
                "instructions": "Take after breakfast"
            }
        ])
        
        # Check what medications are due
        due_medications = []
        upcoming_medications = []
        
        for med in medications:
            for time_str in med.get("schedule", []):
                med_hour = int(time_str.split(":")[0])
                
                # Check if medication is due now (within 1 hour window)
                if abs(current_hour - med_hour) <= 1:
                    due_medications.append({
                        "name": med["name"],
                        "time": time_str,
                        "instructions": med.get("instructions", "")
                    })
                # Check if medication is coming up (next 3 hours)
                elif 0 < med_hour - current_hour <= 3:
                    upcoming_medications.append({
                        "name": med["name"],
                        "time": time_str,
                        "instructions": med.get("instructions", "")
                    })
        
        # Format response
        response_parts = []
        
        if due_medications:
            response_parts.append("**Medications Due Now:**")
            for med in due_medications:
                response_parts.append(f"• {med['name']} - {med['time']}")
                if med['instructions']:
                    response_parts.append(f"  Instructions: {med['instructions']}")
        
        if upcoming_medications:
            response_parts.append("\n**Upcoming Medications:**")
            for med in upcoming_medications:
                response_parts.append(f"• {med['name']} - {med['time']}")
        
        if not due_medications and not upcoming_medications:
            response_parts.append("No medications are due right now. Your next medication will be scheduled later.")
        
        return {
            "formatted_response": "\n".join(response_parts),
            "due_count": len(due_medications),
            "upcoming_count": len(upcoming_medications),
            "medications": {
                "due": due_medications,
                "upcoming": upcoming_medications
            }
        }
    
    def _set_health_reminder(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Set a health reminder"""
        # In production, this would integrate with a scheduling system
        time = context.get("time", "your specified time") if context else "your specified time"
        
        return {
            "formatted_response": (
                f"I'll help you set up that reminder. In the full version, "
                f"I would set a reminder for {time}. For now, here are some tips:\n\n"
                "• Write it down in a visible place\n"
                "• Set an alarm on your phone\n"
                "• Ask a family member to remind you\n"
                "• Use a pill organizer with day labels"
            ),
            "reminder_type": "health",
            "status": "mock_created"
        }
    
    def _provide_health_tip(self, query: str, user_health: Dict[str, Any]) -> Dict[str, Any]:
        """Provide personalized health tips"""
        # General health tips for elderly
        tips = [
            {
                "category": "hydration",
                "tip": "Remember to drink water regularly throughout the day, even if you don't feel thirsty. Aim for 6-8 glasses daily.",
                "why": "Staying hydrated helps maintain kidney function and prevents confusion."
            },
            {
                "category": "exercise",
                "tip": "Try to take a short walk every day, even if it's just around your home or garden.",
                "why": "Daily walking helps maintain strength, balance, and cardiovascular health."
            },
            {
                "category": "sleep",
                "tip": "Keep a regular sleep schedule by going to bed and waking up at the same time each day.",
                "why": "Good sleep helps with memory, mood, and overall health."
            },
            {
                "category": "nutrition",
                "tip": "Include colorful fruits and vegetables in your meals for better nutrition.",
                "why": "A varied diet provides essential vitamins and minerals for healthy aging."
            },
            {
                "category": "social",
                "tip": "Stay connected with friends and family - social interaction is important for mental health.",
                "why": "Regular social contact helps prevent loneliness and keeps your mind active."
            }
        ]
        
        # Select a relevant tip based on query or randomly
        import random
        selected_tip = random.choice(tips)
        
        response = f"**Health Tip: {selected_tip['category'].title()}**\n\n"
        response += f"{selected_tip['tip']}\n\n"
        response += f"💡 Why this matters: {selected_tip['why']}"
        
        return {
            "formatted_response": response,
            "tip_category": selected_tip['category'],
            "tip_content": selected_tip
        }
    
    def _basic_symptom_check(self, query: str) -> Dict[str, Any]:
        """Provide basic symptom information"""
        # Extract symptoms from query
        symptoms = self._extract_symptoms(query)
        
        response_parts = [
            "I understand you're experiencing some discomfort. While I can't provide medical diagnosis, "
            "here's what I recommend:"
        ]
        
        # Check for emergency symptoms
        emergency_symptoms = ["chest pain", "difficulty breathing", "sudden confusion", "severe headache"]
        found_emergency = any(symptom in query.lower() for symptom in emergency_symptoms)
        
        if found_emergency:
            response_parts.append(
                "\n⚠️ **Important**: The symptoms you described could be serious. "
                "Please contact your doctor immediately or call emergency services if needed."
            )
        else:
            response_parts.append("\n**General Advice:**")
            response_parts.append("• Keep track of when symptoms occur and what makes them better or worse")
            response_parts.append("• Make sure you're taking any prescribed medications as directed")
            response_parts.append("• Rest and stay hydrated")
            response_parts.append("• Contact your doctor if symptoms persist or worsen")
        
        response_parts.append(
            "\n📞 **Remember**: Always consult with your healthcare provider for medical concerns."
        )
        
        return {
            "formatted_response": "\n".join(response_parts),
            "symptoms_detected": symptoms,
            "emergency_detected": found_emergency,
            "action_recommended": "seek_immediate_care" if found_emergency else "monitor_and_consult"
        }
    
    def _extract_symptoms(self, query: str) -> List[str]:
        """Extract symptom keywords from query"""
        symptom_keywords = [
            "pain", "ache", "dizzy", "tired", "nausea", "fever",
            "cough", "headache", "weak", "confused", "breathing"
        ]
        
        found_symptoms = []
        query_lower = query.lower()
        
        for symptom in symptom_keywords:
            if symptom in query_lower:
                found_symptoms.append(symptom)
        
        return found_symptoms
    
    def _general_health_response(self, query: str) -> Dict[str, Any]:
        """Provide general health response"""
        return {
            "formatted_response": (
                "I'm here to help with your health questions. You can ask me about:\n\n"
                "• Medication reminders and schedules\n"
                "• General health tips for healthy aging\n"
                "• Setting up health-related reminders\n"
                "• Basic information about symptoms (though always consult your doctor)\n\n"
                "What would you like to know?"
            ),
            "action": "general_info"
        }


# Register the health tool
if __name__ != "__main__":  # Only register when imported
    from src.tools.base import tool_registry
    tool_registry.register(HealthReminderTool())