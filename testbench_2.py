# -*- coding: utf-8 -*-
"""
****************************************************
*             generative_ai_testbench                
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from src.configuration import configuration as cfg
from gtts import gTTS, lang
from playsound import playsound

print(lang.tts_langs())
tmp_path = os.path.join(cfg.PATHS.DATA_PATH, "tts", "tmp")
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)
tmp_path += "/sound.mp3"

s = gTTS("""
Auszug aus "Einführung in das deutsche Recht" von Prof. Dr. Gerhard Robbers:
Das deutsche Recht steht in enger Verknüpfung mit der Gesamtentwicklung europäischer und anglo-amerikanischer Rechtsordnungen. Im Zentrum Europas gelegen, hatDeutschland seit jeher beständigen Austausch der Rechtsideen erlebt. GemeinsameWurzeln und historische Erfahrungen begründen strukturelle Ähnlichkeiten derRechtsordnungen, ihre Unterschiede sind Antworten auf Besonderheiten der politischen Entwicklungen. Vor allem die Einigung Europas in der Europäischen Unionträgt heute dazu bei, Gemeinsamkeiten zu verstärken; es gibt kaum noch ein Rechtsgebiet, das nicht durch Normen des Europäischen Unionsrechts mindestens berührt,wenn nicht geprägt ist. In der institutionellen Ausgestaltung ist das neu, nicht aber inder Sache grundsätzlich gemeinsamer Rechtsentwicklung selbst. Die aus der europäischen Aufklärung des 17. und 18. Jahrhunderts stammende Kodifikationsidee führte zusammen mit den politischen Einigungsbestrebungen des 19. Jahrhunderts zu umfassenden Gesetzbüchern für eine Reihe wichtiger Rechtsgebiete. Siegeben dem deutschen Recht das Gepräge einer kodifizierten, also in umfassenden Gesetzbüchern niedergelegten Rechtsordnung. Es gehört nicht zuletzt deshalb zur kontinentaleuropäischen Rechtsfamilie. Gleichwohl besitzt die Rechtsprechung erheblichesGewicht bei der Fortentwicklung und Konkretisierung des Rechts, so dass der Gegensatz zur anglo-amerikanischen Tradition weniger scharf ist als oft behauptet. Wie die meisten anderen kontinentaleuropäischen Rechtsordnungen ist das deutscheRecht bis heute tief geprägt von der Rezeption des römischen Rechts. In der Spätantikegesammelt, sind große Teile römischer Rechtsregeln zwischen dem 12. und dem 16.Jahrhundert von Oberitalien aus über Europa verbreitet worden. Als eine dogmatischdurchgebildete, schriftlich fixierte Rechtsordnung trat sie neben und oft an die Stelleder überkommenen einzelnen germanischen Stammesrechte. Besonders im 19. Jahrhundert sind diese Überlieferungen systematisiert worden und bilden so eine Grundlage des heute in Deutschland geltenden Rechts. Das gegenwärtige Erscheinungsbild des deutschen Rechts lässt sich nicht verstehen ohne die Katastrophe der nationalsozialistischen Herrschaft zwischen 1933 und 1945.Rechtsetzung und Rechtsanwendung sind in wesentlichen Zügen bis in die Gegenwarthinein von dem Bemühen geprägt, eine Wiederholung solchen Unrechts zu verhindern.Hier besonders liegt begründet, dass das Grundgesetz als Verfassung eines demokratischen und sozialen Rechtsstaates alles andere Recht durchwirkt. Heute ist für dasdeutsche Recht die zentrale Rolle der Grundrechte und des Rechtsstaatsprinzips kennzeichnend. Sie geben der gesamten Rechtsordnung bis in dogmatische Einzelfragenhinein Struktur und Richtung. Dies und das bedeutende Gewicht der Verfassungsgerichtsbarkeit sind wirkkräftiger Teil einer Verrechtlichung auch der Politik. Die einzelnen Motive dieser Entwicklung, mit der weitere Kennzeichen verfassungsstaatlicherOrdnung wie Demokratie, Sozialstaatlichkeit und Bundesstaatlichkeit verwoben sind,wurzeln in älteren, oft in Jahrhunderten gewachsenen Traditionen, die in je unterschiedlicher Ausprägung den europäisch-nordatlantischen Rechtsraum verbinden.
""", lang="de")
s.save(tmp_path)
playsound(tmp_path)
