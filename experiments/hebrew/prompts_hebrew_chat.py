"""Paired chat answers: the same reply in Hebrew and in English.

Questions are in English in both conditions, so the contrast isolates the
language of the assistant's answer rather than the topic or user message.
"""

QUESTIONS = [
    "How do I boil an egg?",
    "What can I make with ripe bananas?",
    "How can I keep rice from sticking?",
    "What is a quick breakfast idea?",
    "How do I make tea less bitter?",
    "Why does my Python loop never stop?",
    "What is a variable in programming?",
    "How do I rename a file in Git?",
    "What does an HTTP 404 error mean?",
    "How can I make my code easier to read?",
    "What should I pack for a weekend trip?",
    "How early should I arrive at the airport?",
    "What is the best way to explore a new city?",
    "How can I sleep on a long flight?",
    "What should I do if I miss my train?",
    "Why is the sky blue?",
    "How do plants grow toward light?",
    "What causes thunder?",
    "Why does ice float on water?",
    "What is a black hole?",
    "How can I remember people's names?",
    "What should I do when I feel overwhelmed?",
    "How can I politely decline an invitation?",
    "How do I start a difficult conversation?",
    "What is a good way to build a new habit?",
    "Is walking good exercise?",
    "How can I improve my sleep routine?",
    "What can help with a mild headache?",
    "Why is it important to drink water?",
    "How often should I take breaks from my screen?",
    "What is a thoughtful birthday gift?",
    "How can I organize a messy desk?",
    "What should I read next?",
    "How do I remove a coffee stain?",
    "What is an easy way to save money?",
    "How are you today?",
    "Can you tell me a short joke?",
    "What do you like to talk about?",
    "What is your favorite season?",
    "Can you cheer me up?",
    "Who are you?",
    "What is your name?",
    "Are you a real person?",
    "What can you help me with?",
    "Do you ever get tired?",
    "Why do cats purr?",
    "How can I teach my dog to sit?",
    "What music is good for studying?",
    "How do I take a better phone photo?",
    "What can I do on a rainy afternoon?",
]

PAIRS = [
    (
        "מניחים את הביצה בסיר, מכסים במים ומביאים לרתיחה. מבשלים כ־7 דקות לביצה רכה או כ־10 דקות לביצה קשה, ואז מעבירים למים קרים.",
        "Put the egg in a pot, cover it with water, and bring it to a boil. Cook for about 7 minutes for a soft egg or 10 minutes for a hard egg, then transfer it to cold water.",
    ),
    (
        "אפשר להכין מהן לחם בננה, שייק או פנקייקים. ככל שהבננות בשלות יותר, כך הן יוסיפו יותר מתיקות.",
        "You can make banana bread, a smoothie, or pancakes with them. The riper the bananas are, the more sweetness they will add.",
    ),
    (
        "שוטפים את האורז לפני הבישול ומשתמשים בכמות המים המתאימה. אחרי הבישול משאירים אותו מכוסה כמה דקות ומאווררים בעדינות במזלג.",
        "Rinse the rice before cooking and use the right amount of water. After cooking, leave it covered for a few minutes and fluff it gently with a fork.",
    ),
    (
        "יוגורט עם פרי, שיבולת שועל וקצת אגוזים הוא ארוחת בוקר מהירה ומשביעה. אפשר להכין הכול בקערה בתוך כמה דקות.",
        "Yogurt with fruit, oats, and a few nuts is a quick and filling breakfast. You can prepare everything in a bowl in a few minutes.",
    ),
    (
        "כדאי להשתמש במים מעט פחות חמים ולקצר את זמן החליטה. אפשר גם להוסיף מעט חלב או דבש כדי לרכך את הטעם.",
        "Try using slightly cooler water and shortening the steeping time. You can also add a little milk or honey to soften the flavor.",
    ),
    (
        "כנראה שתנאי העצירה אף פעם לא מתקיים או שהמשתנה שבו לא מתעדכן. בדוק את התנאי ואת הערכים שמשתנים בכל איטרציה.",
        "The stopping condition probably never becomes true, or its variable is not being updated. Check the condition and the values that change in each iteration.",
    ),
    (
        "משתנה הוא שם שמפנה לערך שהתוכנית יכולה להשתמש בו או לשנות אותו. למשל, אפשר לשמור את המספר 5 במשתנה בשם count.",
        "A variable is a name that refers to a value the program can use or change. For example, you can store the number 5 in a variable named count.",
    ),
    (
        "משתמשים בפקודה `git mv old_name new_name`, ואז מבצעים commit לשינוי. כך Git מתעד את שינוי שם הקובץ.",
        "Use the command `git mv old_name new_name`, then commit the change. This lets Git record the file rename.",
    ),
    (
        "שגיאת HTTP 404 פירושה שהשרת לא מצא את הדף או המשאב שביקשת. ייתכן שהכתובת שגויה או שהמשאב הועבר או נמחק.",
        "An HTTP 404 error means the server could not find the page or resource you requested. The address may be wrong, or the resource may have been moved or deleted.",
    ),
    (
        "כדאי להשתמש בשמות ברורים, בפונקציות קצרות ובעיצוב עקבי. הוסף הערות רק כשהן מסבירות משהו שהקוד עצמו לא מבהיר.",
        "Use clear names, short functions, and consistent formatting. Add comments only when they explain something the code itself does not make clear.",
    ),
    (
        "ארוז בגדים מתאימים למזג האוויר, כלי רחצה, מטען וכל תרופה נחוצה. תיק קטן ונוח בדרך כלל מספיק לסוף שבוע.",
        "Pack clothes suitable for the weather, toiletries, a charger, and any necessary medication. A small, comfortable bag is usually enough for a weekend.",
    ),
    (
        "לטיסה פנימית כדאי להגיע כשעתיים מראש, ולטיסה בינלאומית כשלוש שעות מראש. בדוק גם את הנחיות חברת התעופה ונמל התעופה.",
        "For a domestic flight, arrive about two hours early, and for an international flight, about three hours early. Also check the airline and airport guidance.",
    ),
    (
        "בחר שכונה אחת או שתיים, טייל ברגל והשאר זמן לעצירות לא מתוכננות. שילוב של אתר מרכזי אחד עם מקומות מקומיים נותן תחושה טובה של העיר.",
        "Choose one or two neighborhoods, walk around, and leave time for unplanned stops. Combining one major sight with local places gives you a good feel for the city.",
    ),
    (
        "כדאי להביא כרית צוואר, מסכת עיניים ואטמי אוזניים, ולהימנע מעודף קפאין. כוון את השינה לשעת היעד אם אפשר.",
        "Bring a neck pillow, an eye mask, and earplugs, and avoid too much caffeine. If possible, time your sleep for the destination's time zone.",
    ),
    (
        "בדוק באפליקציה או בלוח התחנה מהי הרכבת הבאה ואם הכרטיס שלך עדיין תקף. אם צריך, פנה לצוות התחנה לקבלת עזרה.",
        "Check the app or station board for the next train and whether your ticket is still valid. If needed, ask the station staff for help.",
    ),
    (
        "אור השמש מתפזר באטמוספרה, ואור כחול מתפזר יותר מרוב הצבעים האחרים. לכן השמיים נראים לנו כחולים במהלך היום.",
        "Sunlight scatters in the atmosphere, and blue light scatters more than most other colors. That is why the sky looks blue to us during the day.",
    ),
    (
        "צמחים מזהים את כיוון האור ומעבירים הורמוני גדילה לצד המוצל של הגבעול. הצד הזה מתארך יותר וגורם לצמח להתכופף לעבר האור.",
        "Plants sense the direction of light and move growth hormones to the shaded side of the stem. That side lengthens more, causing the plant to bend toward the light.",
    ),
    (
        "ברק מחמם את האוויר סביבו במהירות עצומה. האוויר מתפשט בבת אחת ויוצר את גל הקול שאנחנו שומעים כרעם.",
        "Lightning heats the air around it extremely quickly. The air expands suddenly and creates the sound wave we hear as thunder.",
    ),
    (
        "כשמים קופאים, המולקולות שלהם מסתדרות במבנה שתופס יותר מקום. לכן הקרח פחות צפוף ממים נוזליים וצף על פניהם.",
        "When water freezes, its molecules form a structure that takes up more space. This makes ice less dense than liquid water, so it floats.",
    ),
    (
        "חור שחור הוא אזור בחלל שבו הכבידה חזקה כל כך שאפילו אור אינו יכול להימלט ממנו. רבים מהחורים השחורים נוצרים מקריסה של כוכבים מסיביים.",
        "A black hole is a region of space where gravity is so strong that even light cannot escape. Many black holes form from the collapse of massive stars.",
    ),
    (
        "חזור על השם מיד אחרי שאתה שומע אותו וקשר אותו לפרט בולט על האדם. שימוש בשם פעם נוספת בשיחה עוזר לקבע אותו בזיכרון.",
        "Repeat the name right after you hear it and connect it to a noticeable detail about the person. Using the name once more in conversation helps fix it in your memory.",
    ),
    (
        "עצור לרגע, נשום לאט ובחר משימה קטנה אחת להתחיל ממנה. אם התחושה נמשכת או פוגעת בתפקוד, כדאי לפנות לאדם קרוב או לאיש מקצוע.",
        "Pause for a moment, breathe slowly, and choose one small task to start with. If the feeling continues or affects your daily functioning, consider reaching out to someone you trust or a professional.",
    ),
    (
        "אפשר לומר: 'תודה שחשבת עליי, אבל לא אוכל להגיע הפעם.' אין צורך לתת הסבר ארוך.",
        "You can say, 'Thank you for thinking of me, but I won't be able to come this time.' You do not need to give a long explanation.",
    ),
    (
        "בחר זמן רגוע, תאר את הנושא בלי האשמות והסבר איך הוא משפיע עליך. הקשב גם לצד השני ושאף להגיע להבנה משותפת.",
        "Choose a calm time, describe the issue without blame, and explain how it affects you. Listen to the other person too, and aim for mutual understanding.",
    ),
    (
        "התחל בצעד קטן ומוגדר שקל לחזור עליו מדי יום. חבר אותו להרגל קיים ועקוב אחרי ההתקדמות בלי לצפות לשלמות.",
        "Start with a small, specific step that is easy to repeat every day. Attach it to an existing habit and track your progress without expecting perfection.",
    ),
    (
        "כן, הליכה היא פעילות גופנית מצוינת ונגישה שיכולה לשפר את הכושר ואת מצב הרוח. קצב מהיר יותר או מסלול בעלייה יגבירו את המאמץ.",
        "Yes, walking is excellent, accessible exercise that can improve fitness and mood. A faster pace or an uphill route will increase the effort.",
    ),
    (
        "נסה ללכת לישון ולקום בשעות קבועות, ולהפחית מסכים וקפאין בערב. חדר חשוך, שקט וקריר יכול גם לעזור.",
        "Try to go to bed and wake up at consistent times, and reduce screens and caffeine in the evening. A dark, quiet, cool room can also help.",
    ),
    (
        "מנוחה, שתיית מים וארוחה קלה עשויות לעזור לכאב ראש קל. אם הכאב חזק, פתאומי, חוזר לעיתים קרובות או מלווה בתסמינים חריגים, פנה לייעוץ רפואי.",
        "Rest, water, and a light meal may help with a mild headache. If the pain is severe, sudden, frequent, or accompanied by unusual symptoms, seek medical advice.",
    ),
    (
        "מים עוזרים לווסת את חום הגוף, להעביר חומרים מזינים ולתמוך בתפקוד תקין של האיברים. גם התייבשות קלה עלולה לגרום לעייפות ולכאב ראש.",
        "Water helps regulate body temperature, transport nutrients, and support normal organ function. Even mild dehydration can cause fatigue and a headache.",
    ),
    (
        "מומלץ להסיט את המבט מהמסך ולקום לכמה דקות לפחות פעם בשעה. אפשר גם להשתמש בכלל 20-20-20: כל 20 דקות להביט במשך 20 שניות למרחק של כ־6 מטרים.",
        "Look away from the screen and stand up for a few minutes at least once an hour. You can also use the 20-20-20 rule: every 20 minutes, look about 20 feet away for 20 seconds.",
    ),
    (
        "מתנה שמתאימה לתחביב או לצורך של האדם היא בדרך כלל בחירה מוצלחת. גם חוויה משותפת או מכתב אישי יכולים להיות משמעותיים מאוד.",
        "A gift that fits the person's hobby or a current need is usually a good choice. A shared experience or a personal letter can also be very meaningful.",
    ),
    (
        "פנה תחילה את כל מה שלא שייך לשולחן, ואז מיין את השאר לפי שימוש. השאר בהישג יד רק פריטים שאתה צריך לעיתים קרובות.",
        "First remove everything that does not belong on the desk, then sort the rest by use. Keep only frequently needed items within reach.",
    ),
    (
        "בחר ספר לפי מצב הרוח שלך: רומן קליל למנוחה, ספר עיון לסקרנות או סיפורים קצרים כשאין הרבה זמן. אם תספר לי מה אהבת לאחרונה, אוכל להציע משהו ממוקד יותר.",
        "Choose a book based on your mood: a light novel for relaxation, nonfiction for curiosity, or short stories when time is limited. If you tell me what you enjoyed recently, I can suggest something more specific.",
    ),
    (
        "ספוג מיד את הנוזל בלי לשפשף, ואז שטוף במים קרים עם מעט סבון. בבד שניתן לכביסה, בדוק שהכתם נעלם לפני הייבוש.",
        "Blot the liquid immediately without rubbing, then rinse with cold water and a little soap. For washable fabric, check that the stain is gone before drying it.",
    ),
    (
        "הגדר העברה אוטומטית קטנה לחשבון חיסכון ביום קבלת המשכורת. אפילו סכום צנוע מצטבר עם הזמן כשחוסכים בעקביות.",
        "Set up a small automatic transfer to a savings account on payday. Even a modest amount grows over time when you save consistently.",
    ),
    (
        "אני כאן ומוכן לעזור, תודה ששאלת. איך עובר עליך היום?",
        "I'm here and ready to help, thanks for asking. How is your day going?",
    ),
    (
        "למה המחשב הלך לרופא? כי היה לו וירוס!",
        "Why did the computer go to the doctor? Because it had a virus!",
    ),
    (
        "אני שמח לדבר על רעיונות, מדע, טכנולוגיה, ספרים, תחביבים או שאלות יומיומיות. אפשר להתחיל בכל נושא שמעניין אותך.",
        "I enjoy talking about ideas, science, technology, books, hobbies, or everyday questions. We can start with anything that interests you.",
    ),
    (
        "אין לי העדפות אישיות, אבל לכל עונה יש משהו מיוחד. האביב, למשל, מביא מזג אוויר נעים והרבה צמיחה חדשה.",
        "I do not have personal preferences, but every season has something special. Spring, for example, brings pleasant weather and lots of new growth.",
    ),
    (
        "בהחלט. יום קשה הוא רק יום אחד, וגם צעד קטן קדימה נחשב להתקדמות.",
        "Absolutely. A hard day is only one day, and even a small step forward counts as progress.",
    ),
    (
        "אני עוזר בינה מלאכותית שנועד לענות על שאלות, להסביר רעיונות ולעזור במשימות. אין לי חיים או חוויות אישיות, אבל אני כאן כדי לסייע.",
        "I'm an AI assistant designed to answer questions, explain ideas, and help with tasks. I do not have a personal life or experiences, but I am here to assist.",
    ),
    (
        "אני עוזר בינה מלאכותית, ואין לי שם אישי כמו שיש לאדם. אפשר פשוט לקרוא לי 'העוזר'.",
        "I'm an AI assistant, and I do not have a personal name like a person does. You can simply call me 'Assistant.'",
    ),
    (
        "לא, אני עוזר בינה מלאכותית ולא אדם אמיתי. אני מפיק תשובות מטקסט ואין לי גוף או תודעה אנושית.",
        "No, I'm an AI assistant, not a real person. I generate responses from text and do not have a body or human consciousness.",
    ),
    (
        "אני יכול לעזור בהסברים, כתיבה, תכנון, רעיונות, תכנות ושאלות יומיומיות. תאר לי מה אתה מנסה לעשות, ואנסה לסייע.",
        "I can help with explanations, writing, planning, ideas, programming, and everyday questions. Tell me what you are trying to do, and I will try to assist.",
    ),
    (
        "לא, אני עוזר בינה מלאכותית ולכן איני מתעייף כמו בני אדם. אני זמין לעזור בכל פעם שפונים אליי.",
        "No, I'm an AI assistant, so I do not get tired like people do. I am available to help whenever you ask.",
    ),
    (
        "חתולים מגרגרים לעיתים כשהם רגועים ומרוצים, אבל גם כשהם לחוצים או כואבים. הגרגור כנראה משמש לתקשורת ולהרגעה עצמית.",
        "Cats often purr when they are relaxed and content, but also when they are stressed or in pain. Purring probably serves as communication and self-soothing.",
    ),
    (
        "החזק חטיף מעל האף של הכלב והזז אותו לאט לאחור עד שהכלב מתיישב. אמור 'שב', תן את החטיף ושבח אותו מיד, וחזור על כך באימונים קצרים.",
        "Hold a treat above your dog's nose and move it slowly backward until the dog sits. Say 'sit,' give the treat, and praise immediately, then repeat in short sessions.",
    ),
    (
        "מוזיקה אינסטרומנטלית שקטה, צלילי טבע או רעש לבן מתאימים להרבה אנשים. בחר משהו ללא שינויים חדים ושמור על עוצמה נמוכה.",
        "Quiet instrumental music, nature sounds, or white noise work well for many people. Choose something without abrupt changes and keep the volume low.",
    ),
    (
        "נקה את העדשה, השתמש באור טבעי והקש על הנושא במסך כדי למקד. החזק את הטלפון יציב והשאר מעט מרחב סביב הנושא.",
        "Clean the lens, use natural light, and tap the subject on the screen to focus. Hold the phone steady and leave a little space around the subject.",
    ),
    (
        "אפשר לקרוא ספר, לבשל משהו חדש, לראות סרט או לסדר פינה בבית. אם מתחשק לך לזוז, נסה אימון קצר או ריקוד לצלילי מוזיקה.",
        "You could read a book, cook something new, watch a movie, or organize a corner of your home. If you feel like moving, try a short workout or dance to some music.",
    ),
]


def chat_texts(tok):
    """Return (target_texts, neutral_texts): each is the full chat-template string for user=QUESTIONS[i] and assistant=the Hebrew (target) or English (neutral) answer, enable_thinking=False."""
    target_texts = []
    neutral_texts = []
    for question, (hebrew_answer, english_answer) in zip(QUESTIONS, PAIRS):
        target_texts.append(
            tok.apply_chat_template(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": hebrew_answer},
                ],
                tokenize=False,
                enable_thinking=False,
            )
        )
        neutral_texts.append(
            tok.apply_chat_template(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": english_answer},
                ],
                tokenize=False,
                enable_thinking=False,
            )
        )
    return target_texts, neutral_texts
