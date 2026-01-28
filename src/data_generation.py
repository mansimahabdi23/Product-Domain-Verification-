import pandas as pd
import random

DOMAINS = {
   "Finance > Banking": {
       "products": [
           "Personal Loan",
            "Credit Card",
            "Mortgage",
            "Savings Account",
            "Investment Account",
            "Home Loan"
       ]
   },
   "Software > SaaS" :{
       "products": [
           "CRM Software",
           "Accounting Platform",
           "Project Management Tool",
           "Email Marketing Service",
           "Cloud Storage Solution",
           "Collaboration Software",
           "HR Management System",
           "Customer Support Platform",
           "Analytics Dashboard",
           "E-commerce Platform",
           "Inventory Management System"
       ]
   },
   "Agriculture > Equipment" : {
       "products": [
           "Tractor",
           "Plough",
           "Seed Drill Machine",
           "Irrigation System",
           "Harvester",
           "Fertilizer Spreader",
           "Cultivator",
           "Crop Sprayer",
       ]
   },
   "Healthcare > Services" : {
       "products": [
           "Telemedicine Service",
           "Home Healthcare",
           "Medical Billing Service",
           "Health Insurance Plan",
           "Wellness Program",
           "Mental Health Counseling",
           "Physical Therapy Service",
           "Chronic Disease Management",
           "Blood Testing Service",
           "Telemedicine Platform",
           "Nutritional Counseling",
           "Vaccination Clinic",
           "Health Screening Service",
            "Emergency Medical Service"
       ]
   },
   "NGO > Social Services" : {
       "products": [
           "Education Sponsorship Program",
           "Disaster Relief Fund",
           "Community Development Project",
           "Healthcare Outreach Program",
           "Environmental Conservation Initiative",
           "Food Security Program",
           "Child Protection Service",
           "Senior Citizen Support Program",
           "Vocational Training Program",
           "Clean Water Initiative",
           "Homelessness Support Service",
           "Mental Health Awareness Campaign"
       ]
   }
}

samples_per_domain = 50
rows = []


    # for category, samples in DOMAINS.items():
    #     for s in samples:
    #         rows.append({
    #             "product_name": s,
    #             "description": f"{s} designed for professional and commercial use.",
    #             "product_category_tree": category
    #         })

    # df = pd.DataFrame(rows)
    # df.to_csv("data/generated_products.csv", index=False)
    # print("Generated synthetic domain products.")
