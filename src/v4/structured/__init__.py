"""Structured 114-2 course database (data/v4/course_struct.db).

Turns the free-text COURSE columns (note, info, evaluation, schedule, ai_policy, ...)
into rows that search filters and LLM prompts can use directly. Rules first; an LLM
fills only what the rules cannot reach, with every reply cached by input hash.

Modules
  text          cleaning, boilerplate, `|` fragment merge, syllabus language
  scalars       one-row-per-course fields (kind label, general-ed, dept, teaching %, ...)
  sessions      weekly sessions with minutes, plus lab / discussion sessions from notes
  targets       target classes parsed from the unit string
  clauses       note / info -> restriction clauses, info -> change events
  grading       evaluation -> graded items with percentages
  weekly        schedule table -> week, date, topic, hours
  aipolicy      ai_policy category and the conditional-use detail
  aliases       course-name abbreviations (個經, 初會(一), 五管 ...)
  llm           cached Ollama calls for the residue the rules miss
  card          compact LLM-ready course card from structured fields only
  store         SQLite schema and writer
"""
