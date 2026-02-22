---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:2128
- loss:MultipleNegativesRankingLoss
base_model: BAAI/bge-base-en-v1.5
widget:
- source_sentence: '|user|: How can I delete a key?'
  sentences:
  - "\nTo delete the key, the second approver must have Manager access policy for\
    \ the instance or key in order to authorize the key for deletion.\n\n\n\n1. In\
    \ the Keys table of the KMS keys page, you can find keys that are authorized for\
    \ deletion with the following indicators:\n\n\n\n* The Set for deletion column\
    \ has a value of True. The authorization expiration time is displayed in the Deletion\
    \ expiration column.\n* A Trash can icon ![Trash can icon](https://cloud.ibm.com/docs-content/v1/content/217f108cc44e2ce15fb692d4b57b4c628464e908/icons/icon_trash.svg)\
    \ is displayed in the State column. Hover over the icon to view the deletion expiration\
    \ date.\n\n\n\n2. To delete the key, follow the instructions in [Deleting keys\
    \ with the console](https://cloud.ibm.com/docs/hs-crypto?topic=hs-crypto-delete-keysdelete-keys-gui).\n\
    \n\n\nHyper Protect Crypto Services sets a 7-day waiting period that starts after\
    \ you provide the first authorization to delete the key. During this 7-day period,\
    \ the key remains in the [Active state](https://cloud.ibm.com/docs/hs-crypto?topic=hs-crypto-key-states)\
    \ and all key operations are allowed on the key. If no action is taken by the\
    \ second user and the 7-day period expires, you must [restart the dual authorization\
    \ process](https://cloud.ibm.com/docs/hs-crypto?topic=hs-crypto-delete-dual-auth-keysset-key-deletion-api)\
    \ to delete the key.\n\n\n\n\n\n\n\n Authorize deletion for a key with the API\
    \ \n\n\n\n Step 1. Authorize deletion for a key"
  - "Boating Frequently Asked Questions - NYS Parks, Recreation & Historic Preservation\r\
    \n \r\n\tBoating Frequently Asked Questions - NYS Parks, Recreation & Historic\
    \ Preservation\r\n\nNo, the icon can only be placed on licenses and ID's issued\
    \ by the NYS DMV.\nDo I need to notify DMV before they renew my license or ID?\r\
    \n\tNo. DMV will maintain a record of your having completed the boating safety\
    \ course, and automatically place the icon on your license each time it is renewed.\n\
    I have a U.S. Coast Guard License; Can that be listed on the Adventure License?\r\
    \n\tNo, only boating safety certificates can be listed.\nWhat happens if I move\
    \ out of state?\r\n\tIf you move out of state you must relinquish your NY Driver\
    \ license and apply for a driver's license in your new state. In order to maintain\
    \ the Adventure License status you will need to apply for a NY Non-Driver ID which\
    \ will include that designation and maintain it continuously.\nI have a lifetime\
    \ fishing and hunting license and a boating safety certificate. Can I put all\
    \ three on my driver's license?\r\n\tYes you may.\nI live in NY but I have an\
    \ out of state boating safety certificate, do I qualify for the Adventure License\
    \ program?\r\n\tNo, you must have a New York safe boating certificate to be eligible.\n\
    I got my boating safety certificate online and tried to get the Adventure License\
    \ through my provider but I when I click on the link it brings me back to a NYS\
    \ information page. What do I have to do?\r\n\tYou must log into your account\
    \ created with your online provider. For more information, click on the link below\
    \ for your provider and follow the directions on the document that pops up. Alternatively,\
    \ you can bring your printed certificate to any DMV office.\n\nBoat US       Boat-Ed\
    \       \r\n\tBoater Exam       US Power Squadron       \r\n\tAceboater\n\nI paid\
    \ my fee and I have not received my new license/permit/ID yet. Who do I call?\r\
    \n\tYou can check on the mailing status with DMV.\nCustomers who wish to add the\
    \ anchor icon to their enhanced driver's license must have an up-to-date photo\
    \ on file with DMV in order for their request to be processed successfully. If\
    \ your photo is out of date, please"
  - "\nAnalyzing machine learning model performance \n\nReview the annotations that\
    \ were added by the trained model to determine whether any adjustments must be\
    \ made to the model to improve its ability to find valid entity mentions, relation\
    \ mentions, and coreferences in the documents.\n\n\n\n About this task \n\nYou\
    \ can analyze performance by viewing a summary of statistics for entity types,\
    \ relation types, and coreferenced mentions. You can also analyze statistics that\
    \ are presented in a confusion matrix. The confusion matrix helps you compare\
    \ the annotations added by the machine learning model to the annotations in ground\
    \ truth.\n\nThe model statistics provide the following metrics:\n\n\n\n* F1 score\n\
    \nA measurement that considers both precision and recall to compute the score.\
    \ The F1 score can be interpreted as a weighted average of the precision and recall\
    \ values, where an F1 score reaches its best value at 1 and worst value at 0.\
    \ See [Analyzing low F1 scores](https://cloud.ibm.com/docs/watson-knowledge-studio?topic=watson-knowledge-studio-evaluate-mlevaluate-mllowf1).\n\
    * Precision\n\nA measurement that specifies what fraction of the machine learning\
    \ model's output was accurate when compared to the human annotator output. Precision\
    \ is determined by the number of correctly labeled annotations divided by the\
    \ total number of annotations added by the machine learning model. A precision\
    \ score of 1.0 for entity type A means that every mention that was labeled as\
    \ entity type A does indeed belong to that classification. A low precision score\
    \ helps you identify places where the machine learning model created incorrect\
    \ annotations. The score says nothing about how many other mentions that were\
    \ labeled as entity type A by the human annotator were missed by the machine learning\
    \ model; the recall score reflects that information. See [Analyzing low precision\
    \ scores](https://cloud.ibm.com/docs/watson-knowledge-studio?topic=watson-knowledge-studio-evaluate-mlevaluate-mllowp).\n\
    * Recall\n\nA measurement that specifies how many mentions that should have been\
    \ annotated by a given label were actually annotated with that label - the right\
    \ mentions being those that human annotators identified in the same documents."
- source_sentence: '|user|: What size should the bin be for compost?

    |user|: can you please explain the steps for composting at home?

    |user|: compost temperature requirement '
  sentences:
  - 'History of Germany (1945–90) History of Germany (1945–90)

    The conclusion of the final settlement cleared the way for the unification of
    East and West Germany . Formal political union occurred on 3 October 1990 , preceded
    by the GDR declaring its accession to the Federal Republic through Article 23
    of West Germany ''s Basic Law ( meaning that constitutionally , East Germany was
    subsumed into West Germany ) ; but affected in strict legality through the subsequent
    Unification Treaty of 30 August 1990 , which was voted into their constitutions
    by both the West German Bundestag and the East German Volkskammer on 20 September
    1990 . These votes simultaneously extinguished the GDR and affected fundamental
    amendments to the West German Basic Law ( including the repeal of the very Article
    23 under which the GDR had recently declared its post-dated accession ) . On 2
    December 1990 , all - German elections were held for the first time since 1933
    . The `` new '''' country stayed the same as the West German legal system and
    institutions were extended to the east . The unified nation kept the name Bundesrepublik
    Deutschland ( though the simple '' Deutschland '' would become increasingly common
    ) and retained the West German `` Deutsche Mark '''' for currency as well . Berlin
    would formally become the capital of the united Germany , but the political institutions
    remained at Bonn for the time being . Only after a heated 1991 debate did the
    Bundestag conclude on moving itself and most of the government to Berlin as well
    , a process that took until 1999 to complete , when the Bundestag held its first
    session at the reconstructed Reichstag building . Many government departments
    still maintain sizable presences in Bonn as of 2008 .'
  - "Approaches to Composting | US EPA Approaches to Composting | US EPA\nWindrows\
    \ are the most common method of composting in the United States. Windrows can\
    \ be turned periodically or have perforated pipes underneath them to allow air\
    \ flow. Small to large volumes and a wide variety of materials can be composted\
    \ using this method, which makes it adaptable to many scales and sites for composting.\
    \ Windrow composting often requires equipment such as front-end loaders, excavators,\
    \ or windrow turners.\nIn-Vessel\nIn-vessel composting can process a variety of\
    \ organic materials without taking up much space. This method involves feeding\
    \ materials into a vessel, such as a drum, silo, concrete-lined trench, or similar\
    \ enclosed equipment. The materials are then mechanically turned or mixed in the\
    \ vessel with bulking agents like wood chips to ensure aeration. Vessels, also\
    \ known as bioreactors, vary in size and capacity. A similar composting method\
    \ that takes place in an enclosed area is an agitated bay system where mixing\
    \ and aerating equipment moves compost along walled bays or beds as it decomposes\
    \ and stabilizes.\nWorm Composting (Vermicomposting)\n\n Resources\n \n\nCheck\
    \ out EPA’s Home Composting page as well as vermicomposting resources from North\
    \ Carolina State University Cooperative Extension.\n\nWorm composting, or vermicomposting,\
    \ relies on earthworms and microorganisms to break down organic materials into\
    \ vermicompost, or worm castings, a high-quality soil amendment. A vermicomposting\
    \ system can be indoors or outdoors in the shade and sized to match the volume\
    \ of food scraps put into the system. Vermicomposting can used by households,\
    \ schools, farms, institutions, and businesses. A properly maintained vermicomposting\
    \ system does not produce odors or attract pests.\nIt is essential to use a suitable\
    \ worm species, such as red wrigglers (Eisenia fetida). Worms are kept in a closed\
    \ bin or system that should be shielded from light and rain, with sufficient air\
    \ flow, and a way to drain excess liquid. The ideal temperature for a vermicomposting\
    \ system is 55 to 80 degrees Fahrenheit."
  - "Coronavirus FAQs: What Veterans Need To Know | Veterans Affairs Coronavirus FAQs:\
    \ What Veterans Need To Know | Veterans Affairs\nThere is no cure for COVID-19\
    \ at this time. There are treatments for COVID-19, but the risk of severe illness\
    \ and death are still high in certain groups of people. \nTalk to your health\
    \ care provider before taking any medications to prevent or treat COVID-19. \n\
    If you have questions about a treatment, call or send a secure message to your\
    \ VA provider.\nLearn more about COVID-19 treatments on the CDC website\nGet answers\
    \ to COVID-19 treatment questions on the FDA website\n\nAt VA, we've completed\
    \ thousands of research studies with the goal of improving the health and well-being\
    \ of Veterans. Because of this research history, we have the long-term data to\
    \ best understand how illnesses, vaccines, and treatments affect Veterans.\n\n\
    We're now working to find ways to prevent and treat COVID-19 for Veterans and\
    \ others. We have a lot to learn. We can’t learn without doing research. And we\
    \ can’t do research without people who volunteer to participate in research studies. \n\
    \nAs a research study participant, you can help us:\n\nBetter understand how COVID-19\
    \ affects different people\nFind ways to prevent and treat COVID-19 for you, your\
    \ family, and your community\nMake sure vaccines and treatments are safe and effective\
    \ in people of all ages, genders, races, and ethnicities\n\nAs a research participant,\
    \ you may also:\n\nLearn more about your own health\nReceive more regular health\
    \ check-ins as part of your study visits\nGet early access to a new vaccine or\
    \ treatment\n\nLearn more about volunteering for coronavirus research at VA\n\n\
    Health care, appointments, and more\n\nAppointments and health protection levels\n\
    \n What are VA health facilities’ guidelines on masks? \n \n\nWhen you come to\
    \ a VA health facility, your health and safety are our priorities.\nAll facilities\
    \ require masks in high-risk areas like these: \n\nChemotherapy units\nDialysis\
    \ units\nEmergency departments and urgent care \nPost-transplant units \n\nMasks\
    \ are also required for people who are sick with COVID-19 or other viral respiratory\
    \ illnesses. \nYou can choose to wear a mask at any time."
- source_sentence: '|user|: How can I obtain a replacement for my lost New York boating
    safety certificate?

    |user|: do I have to have Boating safety certificates ?

    |user|: I need to register my boat and boat trailer

    |user|: Can I use my NY safety certificate in other states?

    |user|: how much is the class and what type of payment do you accept?

    |user|: do I still need to get a New York State Boating Safety Certificate if
    I have certificate from other parties?'
  sentences:
  - "Boating Frequently Asked Questions - NYS Parks, Recreation & Historic Preservation\r\
    \n \r\n\tBoating Frequently Asked Questions - NYS Parks, Recreation & Historic\
    \ Preservation\r\n\nNo. If you are at least 18 years old, you may rent without\
    \ completing a course. The livery must demonstrate how to properly use the boat\
    \ before renting it. Persons under the age of 18 may only rent a motorboat if\
    \ they hold a \r\n\tsafety certificate.\nI lost my safety certificate. How can\
    \ I get a replacement?\r\n\tYou can order your replacement certificate online\
    \ at https://www.ilostmycard.com/\nWhat if I lost my temporary certificate?\r\n\
    \tTemporary certificates cannot be replaced.\nCan I make a copy of my Boating\
    \ Safety Certificate to keep onboard my boat?\r\n\tNo, you must have the original\
    \ certificate on your person or on board the boat if you are the operator.\nWhy\
    \ haven't I received my safety certificate yet?\r\n\tIt takes approximately 90\
    \ days from your course completion date for your permanent certificate to arrive\
    \ in the mail. Your temporary certificate is valid through those 90 days. If you\
    \ are over 18 years of age, you must purchase your certificate from https://www.ilostmycard.com/\
    \ before it will be mailed to you. If you are within 7 days of your temporary\
    \ certificate’s expiration date and have not yet received your card, please email\
    \ us at boating@parks.ny.gov\nHow can I become an instructor?\r\n\tTo become a\
    \ State certified instructor, you must be at least 18 years of age, be able to\
    \ demonstrate an ability to instruct and must have taken a New York State Parks\
    \ classroom course. For more information, or to begin the certification process,\
    \ click here\nI have a US Coast Guard Auxiliary boating safety certificate. Can\
    \ I exchange it for a New York boating safety certificate?\r\n\tA boating safety\
    \ certificate issued by the US Coast Guard Auxiliary is recognized as a valid\
    \ safe boating certificate – you do not need any further documentation. The same\
    \ is true of certificates issued by the US Power Squadron/America’s Boating Club.\n\
    I took a class online and I have lost my permanent card, can you issue me a replacement\
    \ card?\r\n\tYou need to go to the online provider that issued your card. The\
    \ same is true for lost Power Squadron and Coast Guard Auxiliary cards. NYS can\
    \ only issue replacement certificates for the NY Safe Boating Course."
  - 'Galaxies - NASA Science Galaxies - NASA Science

    TechnologyTechnology Living in SpaceManufacturing and MaterialsRoboticsScience
    InstrumentsComputingThe Universe ExoplanetsThe Search for Life in the UniverseStarsGalaxiesBlack
    HolesThe Big BangDark Matter &amp; Dark EnergyThe Solar System The SunMercuryVenusEarthThe
    MoonMarsJupiterSaturnUranusNeptunePluto &amp; Dwarf PlanetsAsteroids, Comets &amp;
    MeteorsThe Kuiper BeltThe Oort CloudSkywatchingEspañol CienciaAeronáuticaCiencias
    TerrestresSistema SolarUniversoScience All NASA ScienceEarth SciencePlanetary
    ScienceAstrophysics & Space ScienceThe Sun & HeliophysicsBiological & Physical
    SciencesLunar ScienceCitizen ScienceAstromaterialsAeronautics ResearchHuman Space
    Travel ResearchExploreSearchSubmitNews & EventsMultimediaNASA+UniverseCosmic HistoryBuilding
    BlocksForcesGalaxiesOverviewTypesEvolutionLarge Scale StructuresBlack HolesOverviewTypesAnatomyBlack
    Hole WeekStarsOverviewTypesMultiple Star SystemsPlanetary SystemsExoplanetsExplorationSensing
    the UniverseTelescopes 101ObservatoriesMoreNewsDeep DivesQuick ReadsMultimediaGlossaryExplore
    This SectionUniverseGalaxiesBlack HolesStarsExoplanetsExplorationMoreGalaxy BasicsGalaxies
    consist of stars, planets, and vast clouds of gas and dust, all bound together
    by gravity. The largest contain trillions of stars and can be more than a million
    light-years across. The smallest can contain a few thousand stars and span just
    a few hundred light-years. Most large galaxies have supermassive black holes at
    their centers, some with billions of times the Sun’s mass.Galaxies come in a variety
    of shapes, mostly spirals and ellipticals, as well as those with less orderly
    appearances, usually dubbed irregular.Most galaxies are between 10 billion and
    13.6 billion years old. Some are almost as old as the universe itself, which formed
    around 13.8 billion years ago. Astronomers think the youngest known galaxy formed
    approximately 500 million years ago.Galaxies can organize into groups of about
    100 or fewer members held together by their mutual gravity.'
  - '

    The safest place to put money is a mixture of cash, local municipal bond funds
    with average durations under two years and US Treasury bond funds with short durations.  Examples
    of good short term US municipal funds: I''m not an active investor in Australian
    securities, so I won''t recommend anything specific. Because rates are so low
    right now, you want a short duration (ie. funds where the average bond matures
    in < 2 years) fund to protect against increased rates. The problem with safety
    is that you won''t make any money. If your goal to grow the value of your investment
    while minimizing risk, you need to look at equities. The portfolios posted by
    justkt are a great place to start.'
- source_sentence: '|user|: As an employee, when is it inappropriate to request to
    see your young/startup company''s financial statements?

    |user|: what are the risks of working at a startup?

    |user|: tips for starting my own

    |user|: that sounds like a lot of work maybe I should just invest in existing
    startups

    |user|: but is it ok if I only can invest a little?

    |user|: Am I better off going to work for a FAANG?

    |user|: which FAANG pays the most?

    |user|: what about goldman sachs or other finance companies'
  sentences:
  - 'Global Warming Global Warming

    Volcanic activity has also, in the deep past, increased greenhouse gases over
    millions of years, contributing to episodes of global warming.

    A biographical sketch of Milutin Milankovitch describes how changes in Earth’s
    orbit affects its climate.

    These natural causes are still in play today, but their influence is too small
    or they occur too slowly to explain the rapid warming seen in recent decades.
    We know this because scientists closely monitor the natural and human activities
    that influence climate with a fleet of satellites and surface instruments.


    Remote meteorological stations (left) and orbiting satellites (right) help scientists
    monitor the causes and effects of global warming. [Images courtesy NOAA Network
    for the Detection of Atmospheric Composition Change (left) and Environmental Visualization
    Laboratory (right).]


    NASA satellites record a host of vital signs including atmospheric aerosols (particles
    from both natural sources and human activities, such as factories, fires, deserts,
    and erupting volcanoes), atmospheric gases (including greenhouse gases), energy
    radiated from Earth’s surface and the Sun, ocean surface temperature changes,
    global sea level, the extent of ice sheets, glaciers and sea ice, plant growth,
    rainfall, cloud structure, and more.

    On the ground, many agencies and nations support networks of weather and climate-monitoring
    stations that maintain temperature, rainfall, and snow depth records, and buoys
    that measure surface water and deep ocean temperatures. Taken together, these
    measurements provide an ever-improving record of both natural events and human
    activity for the past 150 years.

    Scientists integrate these measurements into climate models to recreate temperatures
    recorded over the past 150 years. Climate model simulations that consider only
    natural solar variability and volcanic aerosols since 1750—omitting observed increases
    in greenhouse gases—are able to fit the observations of global temperatures only
    up until about 1950. After that point, the decadal trend in global surface warming
    cannot be explained without including the contribution of the greenhouse gases
    added by humans.

    Though people have had the largest impact on our climate since 1950, natural changes
    to Earth’s climate have also occurred in recent times. For example, two major
    volcanic eruptions, El Chichon in 1982 and Pinatubo in 1991, pumped sulfur dioxide
    gas high into the atmosphere. The gas was converted into tiny particles that lingered
    for more than a year, reflecting sunlight and shading Earth’s surface.'
  - '

    Today typically a Business needs to hold accounts in more than one currency. Banks
    in certain countries are offering what is called a dual currency account. It is
    essentially 2 accounts with same account number but different currency. So One
    can have an account number say 123456 and have it in say AUD and USD. So the balance
    will always show as X AUD and Y USD. If you deposit funds [electronic, check or
    cash] in USD; your USD balance goes up. Likewise at the time of withdrawal you
    have to specify what currency you are withdrawing. Interest rates are calculated
    at different percentage for different currencies.  So in a nutshell it would like
    operating 2 accounts, with the advantage of remembering only one account number.
    Designate a particular currency as default currency. So if you don''t quote a
    currency along with the account number, it would be treated as default currency.
    Otherwise you always quote the account number and currency. Of-course bundled
    with other services like free Fx Advice etc it makes the entire proposition very
    attractive. Edit: If you have AUD 100 and USD 100, if you try and withdraw USD
    110, it will not be allowed; Unless you also sign up for a auto sweep conversion.  If
    you deposit a GBP check into the account, by default it would get converted into
    AUD [assuming AUD is the default currency]'
  - '

    MBA here. I had a relatively well-paying job with great benefits and tuition reimbursement.
    I ended up leaving it for a much lower-paying job though.  Why did I do that?
    I was miserable. The job was to do only as I was told. Also, they changed my project
    scope to something completely different after 4 months. I also was in a terrible
    city with no friends. I hated my job and eventually I hated my life.  I''ve since
    moved back to where I went to college.  I hate that business publications talk
    about MBA''s like we''re all mindless robots looking for the highest paycheck
    at Goldman Sachs. I''ll gladly take friends, a better location, and a comfortable
    salary over anything else.'
- source_sentence: '|user|: How do I find specific conversations?

    |user|: Is a user ID necessary to find a specific conversation?

    |user|: How to create a user ID?

    |user|: What is a strong password?

    |user|: Retention time for conversations?'
  sentences:
  - '

    The statistics can cover a longer time period than the period for which logs of
    conversations are retained.


    ![Time period control](https://cloud.ibm.com/docs-content/v1/content/417db917d18067d9dcc4cf2612564a9a87c3ddc8/assistant/images/oview-time.png)


    You can choose whether to view data for a single day, a week, a month, or a quarter.
    In each case, the data points on the graph adjust to an appropriate measurement
    period. For example, when viewing a graph for a day, the data is presented in
    hourly values, but when viewing a graph for a week, the data is shown by day.
    A week always runs from Sunday through Saturday.


    You can create custom time periods also, such as a week that runs from Thursday
    to the following Wednesday, or a month that begins on any date other than the
    first.


    The time shown for each conversation is localized to reflect the time zone of
    your browser. However, API log calls are always shown in UTC time. As a result,
    if you choose a single day view, for example, the time shown in the visualization
    might differ from the timestamp specified in the log for the same conversation.


    ![Time period control](https://cloud.ibm.com/docs-content/v1/content/417db917d18067d9dcc4cf2612564a9a87c3ddc8/assistant/images/oview-time2.png)

    * Intents and Entities filters - Use either of these drop-down filters to show
    data for a specific intent or entity in your skill.


    The intent and entities filters are populated by the intents and entities in the
    skill, and not what is in the data source. If you have [selected a data source](https://cloud.ibm.com/docs/assistant?topic=assistant-logslogs-deploy-id)
    other than the skill, you might not see an intent or entity from your data source
    logs as an option in the filters, unless those intents and entities are also in
    the skill.'
  - 'Eating disorder Eating disorder

    In addition to socioeconomic status being considered a cultural risk factor so
    is the world of sports . Athletes and eating disorders tend to go hand in hand
    , especially the sports where weight is a competitive factor . Gymnastics , horse
    back riding , wrestling , body building , and dancing are just a few that fall
    into this category of weight dependent sports . Eating disorders among individuals
    that participate in competitive activities , especially women , often lead to
    having physical and biological changes related to their weight that often mimic
    prepubescent stages . Oftentimes as women ''s bodies change they lose their competitive
    edge which leads them to taking extreme measures to maintain their younger body
    shape . Men often struggle with binge eating followed by excessive exercise while
    focusing on building muscle rather than losing fat , but this goal of gaining
    muscle is just as much an eating disorder as obsessing over thinness . The following
    statistics taken from Susan Nolen - Hoeksema ''s book , ( ab ) normal psychology
    , shows the estimated percentage of athletes that struggle with eating disorders
    based on the category of sport .'
  - 'NASA - NSSDCA - Spacecraft - Details NASA - NSSDCA - Spacecraft - Details

    NASA - NSSDCA - Spacecraft - Details


    Friday, 22 March 2024


    Deep Impact/EPOXINSSDCA/COSPAR ID: 2005-001ADescriptionThe goals of the Deep Impact
    mission were to rendezvous with comet 9P/Tempel 1 and launch a projectile into
    the comet nucleus. Observations were made of the ejecta, much of which represented
    pristine material from the interior of the comet, the crater formation process,
    the resulting crater, and outgassing from the nucleus, particularly the newly
    exposed surface. The scientific objectives of the mission are to: improve the
    knowledge of the physical characteristics of cometary nuclei and directly assess
    the interior of cometary nucleus; determine properties of the surface layers such
    as density, strength, porosity, and composition from the crater and its formation;
    study the relationship between the surface layers of a cometary nucleus and the
    possibly pristine materials of the interior by comparison of the interior of the
    crater with the surface before impact; and improve our understanding of the evolution
    of cometary nuclei, particularly their approach to dormancy, by comparing the
    interior and the surface. This project was selected as a Discovery class mission
    in July, 1999. After the primary mission, Deep Impact was selected for a two-part
    extended mission designated EPOXI.


    Spacecraft and Subsystems


    The spacecraft consists of a 370 kg cylindrical copper impactor attached to a
    650 kg flyby bus. The spacecraft is a box-shaped honeycomb aluminum framework
    with a flat rectangular Whipple debris shield mounted on one side to protect components
    during comet close approach. Body mounted on the framework are one high- and one
    medium-resolution instrument, each of which consists of an imaging camera and
    an infrared spectrometer which will be used to observe the ejected ice and dust,
    much of which will be exposed to space for the first time in over 4 billion years.
    The medium resolution camera has a field of view (FOV) of 0.587 degrees and a
    resolution of 7 m/pixel at 700 km distance and is used for navigation and context
    images. The high resolution camera has a FOV of 0.118 degrees and a resolution
    of 1.4 m/pixel at 700 km.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on BAAI/bge-base-en-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) <!-- at revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': True, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '|user|: How do I find specific conversations?\n|user|: Is a user ID necessary to find a specific conversation?\n|user|: How to create a user ID?\n|user|: What is a strong password?\n|user|: Retention time for conversations?',
    '\nThe statistics can cover a longer time period than the period for which logs of conversations are retained.\n\n![Time period control](https://cloud.ibm.com/docs-content/v1/content/417db917d18067d9dcc4cf2612564a9a87c3ddc8/assistant/images/oview-time.png)\n\nYou can choose whether to view data for a single day, a week, a month, or a quarter. In each case, the data points on the graph adjust to an appropriate measurement period. For example, when viewing a graph for a day, the data is presented in hourly values, but when viewing a graph for a week, the data is shown by day. A week always runs from Sunday through Saturday.\n\nYou can create custom time periods also, such as a week that runs from Thursday to the following Wednesday, or a month that begins on any date other than the first.\n\nThe time shown for each conversation is localized to reflect the time zone of your browser. However, API log calls are always shown in UTC time. As a result, if you choose a single day view, for example, the time shown in the visualization might differ from the timestamp specified in the log for the same conversation.\n\n![Time period control](https://cloud.ibm.com/docs-content/v1/content/417db917d18067d9dcc4cf2612564a9a87c3ddc8/assistant/images/oview-time2.png)\n* Intents and Entities filters - Use either of these drop-down filters to show data for a specific intent or entity in your skill.\n\nThe intent and entities filters are populated by the intents and entities in the skill, and not what is in the data source. If you have [selected a data source](https://cloud.ibm.com/docs/assistant?topic=assistant-logslogs-deploy-id) other than the skill, you might not see an intent or entity from your data source logs as an option in the filters, unless those intents and entities are also in the skill.',
    "Eating disorder Eating disorder\nIn addition to socioeconomic status being considered a cultural risk factor so is the world of sports . Athletes and eating disorders tend to go hand in hand , especially the sports where weight is a competitive factor . Gymnastics , horse back riding , wrestling , body building , and dancing are just a few that fall into this category of weight dependent sports . Eating disorders among individuals that participate in competitive activities , especially women , often lead to having physical and biological changes related to their weight that often mimic prepubescent stages . Oftentimes as women 's bodies change they lose their competitive edge which leads them to taking extreme measures to maintain their younger body shape . Men often struggle with binge eating followed by excessive exercise while focusing on building muscle rather than losing fat , but this goal of gaining muscle is just as much an eating disorder as obsessing over thinness . The following statistics taken from Susan Nolen - Hoeksema 's book , ( ab ) normal psychology , shows the estimated percentage of athletes that struggle with eating disorders based on the category of sport .",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.5623, 0.3740],
#         [0.5623, 1.0000, 0.4083],
#         [0.3740, 0.4083, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 2,128 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                           |
  |:--------|:------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                               |
  | details | <ul><li>min: 12 tokens</li><li>mean: 69.88 tokens</li><li>max: 210 tokens</li></ul> | <ul><li>min: 27 tokens</li><li>mean: 307.71 tokens</li><li>max: 512 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>\|user\|: Does being active affect how kids do in school? How?<br>\|user\|: Does physical activity increase a child's attention span?<br>\|user\|: How much physical activity a day is recommended for children?<br>\|user\|: How to avoid child obesity?</code>    | <code>Physical Activity  <br>   <br>	Physical Activity  <br><br>Find support with friends and make it a group activity by scheduling virtual meetups.<br>	<br><br>Learn more<br><br>Exercising with an Injury or Disability<br><br>Limited mobility doesn't mean you can't exercise. Finding ways to stay active and healthy that work for your body are important.<br>	<br><br>Get started<br><br>Fitness During and After Pregnancy<br><br>	Your fitness activities might change while you're pregnant, but staying active is more important than ever. An active lifestyle keeps both you and your baby healthy. And when you’re active, it’s easier to keep up with your kids as they grow!<br>	<br><br>Learn more<br><br>Family Fitness<br><br>Growing children should get at least one hour of physical activity a day. Running, biking, or hiking are all great ways for children to get exercise, but there are many activities children can do indoors as well. Even getting them to assist with the chores can help!<br>	<br><br>Simple exercise for kids<br><br>	Staying active helps kids do better in school. Children who get...</code> |
  | <code>\|user\|: When applying for a mortgage, can it also cover outstanding debts?<br>\|user\|: Is refinancing the same as getting a mortgage?<br>\|user\|: What is the typical down payment amount for a mortgage?<br>\|user\|: Do I need to buy house insurance?</code> | <code><br>They all have rules about what type of insurance you must have;  and if you don't have the required insurance,  they will offer to contract one for you.  That's fair enough,  I had to prove to my CU that the insurance I have was up to their standards,  no biggy.  Where it becomes a scam,  is when i) the lender refuses to accept your existing insurance as valid no matter what;  ii) buys an insurance on your behalf without asking/giving you the option of getting your own;  and iii) buys an overpriced insurance (e.g., insurance for the fair market value of your house rather than the cost of rebuilding it...),  likely from themselves  (and possibly iv) uses this as a pretext to foreclose your house).</code>                                                                                                                                                                                                                                                                                                                                                                                                 |
  | <code>\|user\|: What are the steps to be taken to gather the relevant worker node data?<br>\|user\|: How can i update a classic worker node?<br>\|user\|: Major. menor update.<br>\|user\|: parts of a tag.</code>                                                        | <code><br>Tagging objects in IBM Cloud Object Storage <br><br>Your data can be expressly defined, categorized, and classified in IBM Cloud® Object Storage using associated metadata, called "tags." This document will show you how to take full control in "tagging" the objects representing your data.<br><br><br><br> Objects and metadata <br><br>Organizing your data can be a complex task. Basic methods, such as using key prefixes like organizational "folders" are a great start to hierarchical structures. But for more complex organization, you will need custom "<br><br>tags." Your metadata can describe the relationships inherent to your data, and provide more organization than titles or folders. Unlike mere labels, there are two parts to a tag: a key and a value, defined individually according to your needs.<br><br><br><br> Tagging Objects <br><br>Managing tags describing your objects can be performed through various interfaces and architectures. Using the [Console](https://cloud.ibm.com) provides a graphical user interface. Using the command lin...</code>                                        |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.13.5
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.8.0+cu128
- Accelerate: 1.11.0
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->