def search_hotels(destination: str) -> str:
    """Search for hotels in a destination with realistic recommendations"""
    
    # Hotel database with popular Indian destinations
    hotels_db = {
        "goa": [
            {"name": "Taj Fort Aguada", "price": "₹8,000/night", "rating": "4.5★", "type": "Luxury"},
            {"name": "Lemon Tree Amarante", "price": "₹3,500/night", "rating": "4.2★", "type": "Mid-range"},
            {"name": "Zostel Goa", "price": "₹800/night", "rating": "4.0★", "type": "Hostel"},
        ],
        "jaipur": [
            {"name": "Rambagh Palace", "price": "₹15,000/night", "rating": "4.7★", "type": "Heritage"},
            {"name": "Hotel Pearl Palace", "price": "₹2,500/night", "rating": "4.3★", "type": "Budget"},
            {"name": "Zostel Jaipur", "price": "₹600/night", "rating": "4.1★", "type": "Hostel"},
        ],
        "manali": [
            {"name": "The Himalayan", "price": "₹6,000/night", "rating": "4.4★", "type": "Resort"},
            {"name": "Johnson Lodge", "price": "₹2,000/night", "rating": "4.0★", "type": "Mid-range"},
            {"name": "Backpackers Nest", "price": "₹500/night", "rating": "3.9★", "type": "Hostel"},
        ],
        "delhi": [
            {"name": "The Leela Palace", "price": "₹12,000/night", "rating": "4.6★", "type": "Luxury"},
            {"name": "Hotel Godwin", "price": "₹2,800/night", "rating": "4.2★", "type": "Mid-range"},
            {"name": "Zostel Delhi", "price": "₹700/night", "rating": "4.0★", "type": "Hostel"},
        ],
        "mumbai": [
            {"name": "Taj Mahal Palace", "price": "₹18,000/night", "rating": "4.8★", "type": "Luxury"},
            {"name": "Hotel Suba Palace", "price": "₹3,200/night", "rating": "4.1★", "type": "Mid-range"},
            {"name": "Backpacker Panda", "price": "₹900/night", "rating": "3.8★", "type": "Hostel"},
        ],
        "kerala": [
            {"name": "Kumarakom Lake Resort", "price": "₹10,000/night", "rating": "4.6★", "type": "Resort"},
            {"name": "Spice Village", "price": "₹4,500/night", "rating": "4.3★", "type": "Eco-resort"},
            {"name": "Malabar House", "price": "₹2,000/night", "rating": "4.0★", "type": "Budget"},
        ],
        "udaipur": [
            {"name": "Taj Lake Palace", "price": "₹20,000/night", "rating": "4.9★", "type": "Heritage"},
            {"name": "Jaiwana Haveli", "price": "₹3,000/night", "rating": "4.2★", "type": "Heritage"},
            {"name": "Zostel Udaipur", "price": "₹650/night", "rating": "4.1★", "type": "Hostel"},
        ],
    }
    
    # Normalize destination name
    dest_lower = destination.lower().strip()
    
    # Find matching hotels
    hotels = hotels_db.get(dest_lower)
    
    if hotels:
        result = f"🏨 **Hotels in {destination.title()}:**\n\n"
        for hotel in hotels:
            result += f"• **{hotel['name']}** ({hotel['type']})\n"
            result += f"  💰 {hotel['price']} | {hotel['rating']}\n\n"
        result += "💡 Tip: Book in advance for better rates!"
        return result
    else:
        # Generic response for unlisted destinations
        return f"""🏨 **Hotels in {destination.title()}:**

• **Budget Hotels** (₹1,500-2,500/night)
  Good for backpackers and budget travelers

• **Mid-range Hotels** (₹3,000-5,000/night)
  Comfortable stays with good amenities

• **Luxury Hotels** (₹8,000+/night)
  Premium experience with top facilities

💡 Tip: Check booking.com, MakeMyTrip, or Goibibo for current availability and prices in {destination}."""