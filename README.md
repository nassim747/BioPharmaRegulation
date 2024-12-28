# Magentic One
Magentic One is an open source state-of-the-art multi agents system. With 2 friends, we modified it to adapt it to perform pharmaceutical regulation tasks. Our modifications made it to perform significantly better than the original Magentic One system.

We modified the following files:

  1. multimodal_web_surfer.py
    -A logic-based enhancement was added to detect URLs containing keywords such as "fda.gov," "ema.europa.eu," and others. This allowed the system to directly navigate to these trusted         regulatory sources without additional verification steps, improving efficiency for pharmaceutical-related tasks.

  2. domain_summarizer.py
     -The domain_summarizer.py file was a new addition to the Magentic-One project, created
      specifically to address the need for summarizing regulatory data in a structured and concise
      manner. It focuses on processing text, tables, and images extracted from regulatory
      documents, providing summaries of dosage forms, process specifications, and
      cross-jurisdictional differences.

  3. orchestrator.py
     -A function was added to identify tasks related to regulatory compliance and tailor the orchestrator's behavior accordingly. The initialization process was also updated to include           specific steps for pharmaceutical scenarios, such as utilizing the Domain Summarizer and other agents.
