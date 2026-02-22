---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:2879
- loss:MultipleNegativesRankingLoss
base_model: BAAI/bge-base-en-v1.5
widget:
- source_sentence: '|user|: How do I file a complaint with the state?

    |user|: what if I want to file a complaint with other states

    |user|: what is NY and CA office hours?

    |user|: if I work with New York State Attorney General office, who am I going
    to work with?

    |user|: can you please tell me more about Letitia James?

    |user|: wow, that sounds great. maybe tell me more about Office of the New York
    State Attorney General itself

    |user|: Office of the New York State Attorney General press conference

    |user|: how can I protect myself if I am the whistleblower....'
  sentences:
  - '

    First of all, there are some differences between the retirement accounts that
    you mentioned regarding taxes. Traditional IRA and 401(k) accounts allow you to
    make pre-tax contributions, giving you an immediate tax deduction when you contribute.
    Roth IRA, Roth 401(k) are funded with after tax money, and a non-retirement account
    is, of course, also funded with after tax money. So if you are looking for the
    immediate tax deduction, this is a point in favor of the retirement accounts.
    Roth IRA & Roth 401(k) accounts allow the investment to grow tax-free, which means
    that the growth is not taxed, even when taking the investment out at retirement.
    With Traditional IRA and 401(k) accounts, you need to pay tax on the gains realized
    in the account when you withdraw the money, just as you do with a non-retirement
    account. This is a point in favor of the Roth retirement accounts. To answer your
    question about capital gains, yes, it is true that you do not have a capital gain
    until an investment is sold. So, discounting the contribution tax deductions of
    the retirement accounts, if you only bought individual stocks that never paid
    a dividend, and never sold them until retirement, you are correct that it really
    wouldn''t matter if you had it in a regular brokerage account or in a traditional
    IRA. However, even people dedicated to buy-and-hold rarely actually buy only individual
    stocks and hold them for 30 years. There are several different circumstances that
    will generally happen in the time between now and when you want to withdraw the
    money in retirement that would be taxable events if you are not in a retirement
    account: If you sell an investment and buy a different one, the gains would be
    taxable. If you want to rebalance your holdings, this also involves selling a
    portion of your investments. For example, if you want to maintain an 80% stock/20%
    bond ratio, and your stock values have gone up to 90%, you might want to sell
    some stock and buy bonds. Or if you are getting closer to retirement, you might
    decide to go with a higher percentage of bonds. This would trigger capital gains.
    Inside a mutual fund, anytime the management sells investments inside the fund
    and realizes capital gains, these gains are passed on to the investors, and are
    taxable.'
  - "\nLooking for a fast way to create a cluster from the UI? Try out [Automating\
    \ cluster creation with IBM Cloud Schematics templates](https://cloud.ibm.com/docs/openshift?topic=openshift-templates).\n\
    \n\n\n\n\n Deciding on your cluster setup \n\nAfter you set up your account to\
    \ create clusters, decide on the setup for your cluster. You must make these decisions\
    \ every time that you create a cluster. Review the following decision tree image\
    \ for more information, such as comparisons of Kubernetes and Red Hat OpenShift,\
    \ and VPC and classic infrastructure.\n\n\n\n\n\n Next steps \n\nWhen the cluster\
    \ is up and running, you can check out the following cluster administration tasks:\n\
    \n\n\n* If you created the cluster in a multizone capable zone, [spread worker\
    \ nodes by adding a zone to your cluster](https://cloud.ibm.com/docs/openshift?topic=openshift-add_workers).\n\
    * [Deploy an app in your cluster.](https://cloud.ibm.com/docs/containers?topic=containers-deploy_appapp_cli)\n\
    * [Set up your own private registry in IBM Cloud to store and share Docker images\
    \ with other users.](https://cloud.ibm.com/docs/Registry?topic=Registry-getting-started)\n\
    * [Set up the cluster autoscaler](https://cloud.ibm.com/docs/openshift?topic=openshift-cluster-scaling-classic-vpc)\
    \ to automatically add or remove worker nodes from your worker pools based on\
    \ your workload resource requests.\n* Control who can create pods in your cluster\
    \ with [pod security policies](https://cloud.ibm.com/docs/containers?topic=containers-psp).\n\
    \n\n\nThen, you can check out the following network configuration steps for your\
    \ cluster setup:\n\n\n\n* Classic clusters:\n\n\n\n* Isolate networking workloads\
    \ to edge worker nodes [in classic clusters without a gateway](https://cloud.ibm.com/docs/openshift?topic=openshift-edge)."
  - "Whistleblower Portal | New York State Attorney General Whistleblower Portal |\
    \ New York State Attorney General\nTo learn more about Tor Browser and how it\
    \ works, visit the official Tor Browser online documentation. \n2. Launch Tor\
    \ Browser on your computer.\nAfter installation, open Tor Browser and type the\
    \ following URL into the Tor Browser's address bar:\n\nhttp://wux56igrepfok5cz5eqszt3xr7ng6tqai2ayplbmvp5qxruax5eicjad.onion\n\
    \nIf you are unable to reach the whistleblower portal in Tor Browser, refer to\
    \ Tor Browser's guide to first-time use for additional troubleshooting steps.\n\
    Please note that the URL address above will only work inside Tor Browser.  Attempts\
    \ to visit the URL in your default browser (such as Chrome, Firefox, Safari, or\
    \ Edge) will not connect.\n3. Submit your communication via the whistleblower\
    \ portal.\nThe whistleblower portal consists of a simple form that asks individuals\
    \ to provide the following information:\n\nthe subject of the communication\n\
    a description of the communication\nany supporting files available to upload\n\
    \nOnce the form has been submitted, the submitter will receive a secret code to\
    \ track their submission and communication with our office.\n\nLetitia James\n\
    New York State Attorney General\n\nSocial Media\n\nContact\n\n Office of the New\
    \ York State Attorney General\n The Capitol\n Albany NY 12224-0341\n Phone: 1-800-771-7755\n\
    \nFooter\n\nStatewide Offices\n\nPress Releases\n\nFile a Complaint\n\nEmployment\
    \ Opportunities\n\nCopyright ©\n 2024\n — Office of the New York State Attorney\
    \ General. All Rights Reserved.\n \n\nPolicy Menu\n\nPrivacy Policy\n\nDisclaimer\n\
    \nAccessibility Policy\n\nWe Value Your Privacy\nWe use cookies to enhance your\
    \ browsing experience, serve personalized content, and analyze our traffic. By\
    \ using this website you consent to our use of cookies.\n\nOkay"
- source_sentence: '|user|: What are the steps involved to delete the CIS instance?

    |user|: Do I need to import or recreate my DNS records in CIS?

    |user|: Can I add a custom domain?

    |user|: What is the transportation layer?

    |user|: Is privacy protected?'
  sentences:
  - "\nData privacy and governance \n\nAs a pioneer in the provision of a fully managed\
    \ and globally distributable Database-as-a-Service, IBM® Cloudant® for IBM Cloud®\
    \ allows customers to locate data in any global IBM Cloud® or AWS region. By providing\
    \ customers with such high levels of data mobility to serve the local needs of\
    \ customers, IBM®, and IBM Cloudant take data privacy and governance seriously.\n\
    \nIBM Cloud data privacy processing processes and procedures are documented within\
    \ the IBM Cloud DPA. This Data Processing Addendum (DPA) and its applicable DPA\
    \ Exhibits apply to the Processing of Personal Data by IBM Cloud on behalf of\
    \ Client (Client Personal Data). The processing of Personal Data is subject to\
    \ the General Data Protection Regulation 2016/679 (GDPR). It is also subject to\
    \ any other data protection laws that are identified at [Data Protection Laws](https://www.ibm.com/support/customer/csol/terms?id=DPA-DPL&lc=endetail-document)\
    \ in order to provide services (Services) according to the Agreement between Client\
    \ and IBM Cloud. The IBM Cloud DPA can be found at [Data Processing Addendum](https://www.ibm.com/dpa).\n\
    \nIn addition to the DPA, the cloud services contain DPA exhibits that detail\
    \ the types of data that is processed by this service. Cloud services also contain\
    \ the relevant processing locations (including hosting locations) where client\
    \ PI is processed. The relevant DPA exhibit for IBM Cloudant can be found on the\
    \ [IBM Cloud Terms site](https://www.ibm.com/support/customer/csol/contractexplorer/cloud/datasheets/2052E430379B11E58B2CB2A838CE4F20/en).\n\
    \nIBM Cloud relies on Standard Contractual Clauses (as our primary data transfer\
    \ mechanism) in our customer contracts. IBM Cloud also relies on numerous supplementary\
    \ measures to help clients ensure an adequate level of protection when they transfer\
    \ personal data outside of the EU/EEA."
  - "\n* What options do I have to secure my cluster?\n\nYou can use built-in security\
    \ features in IBM Cloud Kubernetes Service to protect the components in your cluster,\
    \ your data, and app deployments to ensure security compliance and data integrity.\
    \ Use these features to secure your Kubernetes API server, etcd data store, worker\
    \ node, network, storage, images, and deployments against malicious attacks. You\
    \ can also leverage built-in logging and monitoring tools to detect malicious\
    \ attacks and suspicious usage patterns.\n\nFor more information about the components\
    \ of your cluster and how you can meet security standards for each component,\
    \ see [Security for IBM Cloud Kubernetes Service](https://cloud.ibm.com/docs/containers?topic=containers-securitysecurity).\n\
    * What access policies do I give my cluster users?\n\nIBM Cloud Kubernetes Service\
    \ uses Cloud Identity and Access Management (IAM) to grant access to cluster resources\
    \ through IAM platform access roles and Kubernetes role-based access control (RBAC)\
    \ policies through IAM service access roles. For more information about types\
    \ of access policies, see [Pick the correct access policy and role for your users](https://cloud.ibm.com/docs/containers?topic=containers-access-overviewaccess_roles).\n\
    \nThe access policies that you assign users vary depending on what you want your\
    \ users to be able to do. You can find more information about what roles authorize\
    \ which types of actions on the [User access reference page](https://cloud.ibm.com/docs/containers?topic=containers-access_reference)\
    \ or in the following table's links. For steps to assign policies, see [Granting\
    \ users access to your cluster through IBM Cloud IAM](https://cloud.ibm.com/docs/containers?topic=containers-userschecking-perms).\n\
    \n\n\nTypes of roles you might assign to meet different use cases.\n\n Use case\
    \ Example roles and scope"
  - "\nAuditing events for service instances \n\nAs a security officer, auditor, or\
    \ manager, you can use the IBM Cloud Activity Tracker service to track how users\
    \ and applications interact with the IBM Cloud services.\n\nThe IBM Cloud Activity\
    \ Tracker service records user-initiated activities that change the state of a\
    \ service in IBM Cloud. To get started monitoring your user's actions, see [IBM\
    \ Cloud Activity Tracker](https://cloud.ibm.com/docs/services/activity-tracker?topic=activity-tracker-getting-startedgetting-started).\n\
    \n\n\n Events for provisioning and managing service instances \n\nThe following\
    \ table lists the actions that generate an event:\n\n\n\nTable 1. Actions that\
    \ generate events\n\n Action Description \n\n service_name.instance.create An\
    \ event is generated when you provision a service instance. \n service_name.instance.update\
    \ An event is generated when you rename a service instance or when you change\
    \ the service plan. \n service_name.instance.delete An event is generated when\
    \ a service instance is deleted. \n service_name.instance.schedule_reclaim An\
    \ event is generated when a service instance is pending_reclamation. \n service_name.instance.restore\
    \ An event is generated when a service instance is restored. \n\n\n\n\n\n\n\n\
    \ Events for managing aliases that are associated to a service instance \n\nAn\
    \ alias is a connection between your IAM-managed service within a resource group\
    \ and an application within an org or a space.\n\nThe following table lists the\
    \ actions that generate an event:\n\n\n\nTable 2. Actions that generate events\n\
    \n Action Description \n\n service_name.alias.create An event is generated when\
    \ an alias for an instance is created. \n service_name.alias.update An event is\
    \ generated when an alias for an instance is updated. \n service_name.alias.delete\
    \ An event is generated when an alias for an instance is deleted. \n\n\n\n\n\n\
    \n\n Events for managing service credentials that are associated to a service\
    \ instance \n\nA service credential provides the necessary information to connect\
    \ an application to a service instance.\n\nThe following table lists the actions\
    \ that generate an event:\n\n\n\nTable 3. Actions that generate events\n\n Action\
    \ Description \n\n service_name.key.create An event is generated when an API key\
    \ is created for a service instance through the Service credentials section of\
    \ the service instance UI."
- source_sentence: '|user|: What are the steps to be taken to gather the relevant
    worker node data?

    |user|: How can i update a classic worker node?

    |user|: Major. menor update.

    |user|: parts of a tag.

    |user|: node data

    |user|: NodeSync

    |user|: Worker node'
  sentences:
  - '

    I believe the answer is that to protect yourself it is good to get credit protection
    so you will be notified when new credit is taken in your name.   Also, you can
    use http://www.annualcreditreport.com/ to look at your credit report. HINT:  While
    you do that, and while you are in the TransUnion report, you will have the option
    to DISPUTE adverse items.   I always suggest that people dispute everything adverse.  That
    puts the onus on the other parties to produce evidence to TransUnion within 30
    days attesting to the validity of the adverse item.   You would be surprised how
    many will simply drop off your report after doing that.  Everybody should do this
    Here is a direct address for TransUnion: https://dispute.transunion.com/dp/dispute/landingPage.jsp
    ==>  Once the disputes are finalized, the results get communicated to the other
    two bureaus. It is amazing how well it works.  It can raise your credit score
    significantly. It really helps to watch your credit report yourself, and also
    to get whatever protection is offered that may help protect you against others
    opening new accounts in your name.'
  - "\nThe WebSocket interface \n\nThe WebSocket interface of the IBM Watson® Speech\
    \ to Text service is the most natural way for a client to interact with the service.\
    \ To use the WebSocket interface for speech recognition, you first use the /v1/recognize\
    \ method to establish a persistent connection with the service. You then send\
    \ text and binary messages over the connection to initiate and manage recognition\
    \ requests.\n\nBecause of their advantages, WebSockets are the preferred mechanism\
    \ for speech recognition. For more information, see [Advantages of the WebSocket\
    \ interface](https://cloud.ibm.com/docs/speech-to-text?topic=speech-to-text-service-featuresfeatures-websocket-advantages).\
    \ For more information about the WebSocket interface and its parameters, see the\
    \ [API & SDK reference](https://cloud.ibm.com/apidocs/speech-to-text).\n\n\n\n\
    \ Managing a WebSocket connection \n\nThe WebSocket recognition request and response\
    \ cycle has the following steps:\n\n\n\n1. [Open a connection](https://cloud.ibm.com/docs/speech-to-text?topic=speech-to-text-websocketsws-open)\n\
    2. [Initiate a recognition request](https://cloud.ibm.com/docs/speech-to-text?topic=speech-to-text-websocketsws-start)\n\
    3. [Send audio and receive recognition results](https://cloud.ibm.com/docs/speech-to-text?topic=speech-to-text-websocketsws-audio)\n\
    4. [End a recognition request](https://cloud.ibm.com/docs/speech-to-text?topic=speech-to-text-websocketsws-stop)\n\
    5. [Send additional requests and modify request parameters](https://cloud.ibm.com/docs/speech-to-text?topic=speech-to-text-websocketsws-more)\n\
    6."
  - '

    For more information, see [Updating the master](https://cloud.ibm.com/docs/containers?topic=containers-updatemaster).


    Worker nodes in standard clusters are provisioned in to your IBM Cloud infrastructure
    account. The worker nodes are dedicated to your account and you are responsible
    to request timely updates to the worker nodes to ensure that the worker node OS
    and Red Hat OpenShift on IBM Cloud components apply the latest security updates
    and patches. Security updates and patches are made available by IBM Site Reliability
    Engineers (SREs) who continuously monitor the Linux image that is installed on
    your worker nodes to detect vulnerabilities and security compliance issues. For
    more information, see [Updating worker nodes](https://cloud.ibm.com/docs/containers?topic=containers-updateworker_node).

    * Are the master and worker nodes highly available?


    The Red Hat OpenShift on IBM Cloud architecture and infrastructure is designed
    to ensure reliability, low processing latency, and a maximum uptime of the service.
    By default, every cluster in Red Hat OpenShift on IBM Cloud is set up with multiple
    Red Hat OpenShift master instances to ensure availability and accessibility of
    your cluster resources, even if one or more instances of your Red Hat OpenShift
    master are unavailable.


    You can make your cluster even more highly available and protect your app from
    a downtime by spreading your workloads across multiple worker nodes in multiple
    zones of a region. This setup is called a [multizone cluster](https://cloud.ibm.com/docs/openshift?topic=openshift-ha_clustersmz-clusters)
    and ensures that your app is accessible, even if a worker node or an entire zone
    is not available.


    To protect against an entire region failure, create [multiple clusters and spread
    them across IBM Cloud regions](https://cloud.ibm.com/docs/openshift?topic=openshift-ha_clustersmultiple-clusters-glb).
    By setting up a network load balancer (NLB) for your clusters, you can achieve
    cross-region load balancing and cross-region networking for your clusters.'
- source_sentence: '|user|: how can i delete a key?

    |user|: what happen when the first user authorizes a key for deletion?

    |user|: what could happen if no action is taken by the second user and the 7-day
    period experies?

    |user|: in case, the server expired, could i remove the entry from the resource
    list?

    |user|: why could not i delete keys?

    |user|: what could happen if i force a deletion on a key?

    |user|:  cryptoshredding .'
  sentences:
  - "Climate Change - NYSDEC Climate Change - NYSDEC\nSome key points from these reports\
    \ are:Human influence has rapidly warmed the climate at a rate that is unprecedented\
    \ in at least the last 2,000 years.Global warming is driven by emissions from\
    \ human activities.Climate change is already affecting every inhabited region\
    \ across the globe with human influence contributing to many observed changes\
    \ in weather and climate extremes.Limiting human-induced global warming will require\
    \ rapidly reducing and then eliminating GHG emissions and removing emissions previously\
    \ trapped in the atmosphere.Current efforts to reduce GHG emissions will not be\
    \ able to prevent all global warming, so prioritizing resilience and adaptation\
    \ to some inevitable climate change is essential as climate change impacts continue\
    \ to happen and are anticipated to worsen around the globe and regionally.Studying\
    \ Climate Change in the Social SciencesThousands of studies by researchers around\
    \ the world have shown rising air, land, and ocean temperatures, and how the climate\
    \ has been changing over the last century. Despite the amount of available scientific\
    \ knowledge about the impacts of climate change happening now, and of new risks\
    \ for the future, there is still a lack of understanding amongst the general public\
    \ on the subject. Social scientists have been examining the attitudes that people\
    \ have towards climate change. A 2017 Yale study found that 70 percent of the\
    \ U.S. acknowledges that climate change is happening, with varying degrees of\
    \ concern. But knowing that climate change is happening does not necessarily mean\
    \ that people understand how human activities are contributing to GHG emissions\
    \ and climate change, nor does it mean that people are able to easily adopt low-carbon\
    \ lifestyles. Presenting climate change science clearly, in terms that the public\
    \ can relate to, and informed by the insights of social and behavioral science,\
    \ is an important focus and challenge for climate change communicators and educators.\
    \ Communicating the options and benefits of low-carbon best practices is also\
    \ important.\n\nClimate Change Science in New York State\n\n Sea level rise, heatwaves,\
    \ floods, and more frequent storms are just a few climate change impacts that\
    \ New York is experiencing. For more information on New York climate change data,\
    \ indicators, and mapping tools see the additional resources listed below.\n \n\
    \ \n\nAdditional Resources\n\nClimate Change Indicators in the United States\n\
    \nCoastal NY Future Floodplain Mapper\n\nNOAA Sea Level Rise Viewer\n\nThis Page\
    \ Covers\n\nNew York State\n\nIcon \n\nDepartment of Environmental Conservation\n\
    \nQuick Links\n\nAbout DEC"
  - '

    Some things you can do to reduce your expenses - make coffees and lunches at home
    before going out and buying these, pay off higher interest debts first, consolidate
    all your debts into a lower interest rate loan, reduce discretionary spending
    to an absolute minimum, cancel all unnecessary services, etc. Debt Consolidation
    In regards to a Debt Consolidation for your existing personal loans and credit
    cards into a single lower interest rate loan can be a good idea, but there are
    some pitfalls you should consider. Manly, if you are taking out a loan with a
    lower interest rate but a longer term to pay it off, you may end up paying less
    in monthly repayments but will end up paying more interest in the long run. If
    you do take this course of action try to keep your term to no longer than your
    current debt''s terms, and try to keep your repayments as high as possible to
    pay the debt off as soon as possible and reduce any interest you have to pay.
    Again be wary of the fine print and read the PDS of any products you are thinking
    of getting. Refer to ASIC - Money Smart website for more valuable information
    you should consider before taking out any debt consolidation. Assistance improving
    your skills and getting a higher paid job If you are finding it hard to get a
    job, especially one that pays a bit more, look into your options of doing a course
    and improving your skills. There is plenty of assistance available for those wanting
    to improve their skills in order to improve their chances of getting a better
    job. Check out Centrelink''s website for more information on Payments for students
    and trainees. Other Action You Can Take If you are finding that the repayments
    are really getting out of hand and no one will help you with any debt consolidation
    or reducing your interest rates on your debts, as a last resort you can apply
    for a Part 9 debt agreement. But be very careful as this is an alternative to
    bankruptcy, and like bankruptcy a debt agreement will appear on your credit file
    for seven years and your name will be listed on the National Personal Insolvency
    Index forever. Further Assistance and Help If you have trouble reading any PDS,
    or want further information or help regarding any issues I have raised or any
    other part of your financial situation you can contact Centrelink''s Financial
    Information Service. They provide a free and confidential service that provides
    education and information on financial and lifestyle issues to all Australians.'
  - "\nLaunch mysqladmin like this:\n\nmysqladmin [options] command [command-arg]\
    \ [command command-arg]] ...\n\n\n\n\n\n Cryptoshredding keys \n\nKey Protect\
    \ provides for a [force delete](https://cloud.ibm.com/docs/key-protect?topic=key-protect-delete-keys)\
    \ of a key that is in use by IBM Cloud® services, including your Cloud Databases\
    \ deployments. This action is called cryptoshredding.\n\nCryptoshredding is a\
    \ destructive action. When the key is deleted, your data is unrecoverable even\
    \ from a soft delete state.\n\n\n\n\n\n Backups Removal \n\nBackups cannot be\
    \ manually deleted. However, if you delete your deployment, its backups are deleted\
    \ automatically.\n\n\n\n\n\n Reenabling from a soft delete \n\nYou are able to\
    \ discover available soft-deleted instances by using the IBM Cloud CLI [ibmcloud\
    \ resource reclamations](https://cloud.ibm.com/docs/cli?topic=cli-ibmcloud_commands_resourceibmcloud_resource_reclamations)\
    \ command.\n\nYou can then \"undelete\", recover, or reclaim an available soft-deleted\
    \ instance by using the IBM Cloud CLI [ibmcloud resource reclamation-restore](https://cloud.ibm.com/docs/cli?topic=cli-ibmcloud_commands_resourceibmcloud_resource_reclamation_restore)\
    \ command:\n\nibmcloud resource reclamation-restore resource_ID"
- source_sentence: '|user|: is dynamic secret better than static secret?'
  sentences:
  - "Relay: A Laser-Based Space Communications Game | NASA Space Place – NASA Science\
    \ for Kids Relay: A Laser-Based Space Communications Game | NASA Space Place –\
    \ NASA Science for Kids\n\n\nRelay: A Laser-Based Space Communications Game |\
    \ NASA Space Place – NASA Science for Kids\n\nrelay-laser-communications-game\n\
    \nEarth\nSun\nSolar System\nUniverse\nScience and Tech\nEducators\n\nRelay: A\
    \ Laser-Based Space Communications Game\n\nAbout the game\nIn this game you will\
    \ send information, or data, back to Earth using lasers. You will use mirrors\
    \ to move the laser and avoid obstacles to get the data back to Earth. Each level\
    \ has new challenges and your ultimate goal is to build a communication network\
    \ and synchronize, or match, the time on atomic clocks.\n\nAbout the mission\n\
    Optical communication uses lasers to send data with a type of light that we can’t\
    \ see, called infrared light. Since the 1950s, NASA missions have used a different\
    \ type of waves, called radio waves, to send data to and from space. With current\
    \ radio wave communications, it takes about nine weeks to send a complete map\
    \ of Mars back to Earth. With lasers, it only takes about nine days!\nOptical\
    \ communication is very useful for space missions because data can be sent faster\
    \ and more securely. Optical communication systems are also smaller, weigh less,\
    \ and use less power. A smaller size means more room for science instruments,\
    \ and using less power means less of a drain of the spacecraft’s batteries. These\
    \ are all very important details for NASA when designing and creating mission\
    \ concepts. To learn more, visit Laser Communications | NASA.\n\n        \tarticle\
    \ last updated July 28, 2022\n        \n\nIf you liked this, you may like:\n\n\
    DSN Uplink-Downlink: A DSN Game\n\nSnap it! An Eclipse Photo Adventure\n\nExplore\
    \ Mars: A Mars Rover Game\n\n\t    Games\t    \n\n\t    Crafts\t    \n\n\t   \
    \ Activities\t    \n\n\t    Videos\t    \n\n\t    Glossary\t    \n\n\t    Mystery\t\
    \    \n\nAbout Us\nPrivacy Policy\nImage Use\nAccessibility\n\nContact NASA Space\
    \ Place\n\n        Last Updated: May 6th, 2024    \n\n \n\nMore\nLess\n\nBy Subject\n\
    \nSpace\nSun\nEarth\nSolar System\nPeople & Technology\nParents & Educators\n\n\
    By Type\n\nExplore\nDo\nPlay\n\nMore\nLess"
  - 'History of the Great Wall of China History of the Great Wall of China

    The settlement of the north continued up to Qin Shi Huang ''s death in 210 BC
    , upon which Meng Tian was ordered to commit suicide in a succession conspiracy
    . Before killing himself , Meng Tian expressed regret for his walls : `` Beginning
    at Lintao and reaching to Liaodong , I built walls and dug moats for more than
    ten thousand li ; was it not inevitable that I broke the earth ''s veins along
    the way ? This then was my offense . '''''
  - '

    A script and tool will assist you to simulate the transmission of web server log
    messages from a static file to Event Streams.


    Object Storage Event Streams


    +3


    Analytics Engine,Data Engine,Key Protect




    * 3 hours

    * 2023-05-05




    [Getting started with Secrets Manager](https://cloud.ibm.com/docs/secrets-manager?topic=secrets-manager-getting-started)Getting
    started with Secrets Manager


    This tutorial focuses on storing and managing a username and password in IBM Cloud®
    Secrets Manager. With Secrets Manager, you can create, lease, and centrally manage
    secrets that are used in IBM Cloud services or your custom-built applications.
    Secrets are stored in a dedicated Secrets Manager instance, built on open source
    HashiCorp Vault.


    Secrets Manager




    * 10 minutes

    * 2023-03-01




    [Access a storage bucket by using a dynamic secret](https://cloud.ibm.com/docs/secrets-manager?topic=secrets-manager-tutorial-access-storage-bucket)Access
    a storage bucket by using a dynamic secret


    In this tutorial, you learn how to use IBM Cloud® Secrets Manager to create and
    lease an IAM credential that can be used to access a bucket in Cloud Object Storage.


    Secrets Manager Object Storage




    * 1 hour

    * 2023-05-30




    [Secure secrets for apps that run in your Kubernetes cluster](https://cloud.ibm.com/docs/secrets-manager?topic=secrets-manager-tutorial-kubernetes-secrets)Secure
    secrets for apps that run in your Kubernetes cluster


    This tutorial is for the Classic flavor of Kubernetes Service clusters. External
    Secrets is also available as an OpenShift operator.


    Secrets Manager




    * 45 minutes

    * 2023-06-15




    [Part 2: Create a GitHub issue when your certificates are about to expire](https://cloud.ibm.com/docs/secrets-manager?topic=secrets-manager-tutorial-expiring-secrets-part-2)Part
    2: Create a GitHub issue when your certificates are about to expire'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on BAAI/bge-base-en-v1.5
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: validation
      type: validation
    metrics:
    - type: cosine_accuracy@1
      value: 0.28448275862068967
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.47413793103448276
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.5431034482758621
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.6206896551724138
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.28448275862068967
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.23275862068965514
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.17758620689655175
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.11637931034482758
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.11336206896551723
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.2737068965517241
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.33821839080459765
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.43893678160919536
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.3512575534625978
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.39615489874110565
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.29610670488268165
      name: Cosine Map@100
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
    '|user|: is dynamic secret better than static secret?',
    '\nA script and tool will assist you to simulate the transmission of web server log messages from a static file to Event Streams.\n\nObject Storage Event Streams\n\n+3\n\nAnalytics Engine,Data Engine,Key Protect\n\n\n\n* 3 hours\n* 2023-05-05\n\n\n\n[Getting started with Secrets Manager](https://cloud.ibm.com/docs/secrets-manager?topic=secrets-manager-getting-started)Getting started with Secrets Manager\n\nThis tutorial focuses on storing and managing a username and password in IBM Cloud® Secrets Manager. With Secrets Manager, you can create, lease, and centrally manage secrets that are used in IBM Cloud services or your custom-built applications. Secrets are stored in a dedicated Secrets Manager instance, built on open source HashiCorp Vault.\n\nSecrets Manager\n\n\n\n* 10 minutes\n* 2023-03-01\n\n\n\n[Access a storage bucket by using a dynamic secret](https://cloud.ibm.com/docs/secrets-manager?topic=secrets-manager-tutorial-access-storage-bucket)Access a storage bucket by using a dynamic secret\n\nIn this tutorial, you learn how to use IBM Cloud® Secrets Manager to create and lease an IAM credential that can be used to access a bucket in Cloud Object Storage.\n\nSecrets Manager Object Storage\n\n\n\n* 1 hour\n* 2023-05-30\n\n\n\n[Secure secrets for apps that run in your Kubernetes cluster](https://cloud.ibm.com/docs/secrets-manager?topic=secrets-manager-tutorial-kubernetes-secrets)Secure secrets for apps that run in your Kubernetes cluster\n\nThis tutorial is for the Classic flavor of Kubernetes Service clusters. External Secrets is also available as an OpenShift operator.\n\nSecrets Manager\n\n\n\n* 45 minutes\n* 2023-06-15\n\n\n\n[Part 2: Create a GitHub issue when your certificates are about to expire](https://cloud.ibm.com/docs/secrets-manager?topic=secrets-manager-tutorial-expiring-secrets-part-2)Part 2: Create a GitHub issue when your certificates are about to expire',
    'Relay: A Laser-Based Space Communications Game | NASA Space Place – NASA Science for Kids Relay: A Laser-Based Space Communications Game | NASA Space Place – NASA Science for Kids\n\n\nRelay: A Laser-Based Space Communications Game | NASA Space Place – NASA Science for Kids\n\nrelay-laser-communications-game\n\nEarth\nSun\nSolar System\nUniverse\nScience and Tech\nEducators\n\nRelay: A Laser-Based Space Communications Game\n\nAbout the game\nIn this game you will send information, or data, back to Earth using lasers. You will use mirrors to move the laser and avoid obstacles to get the data back to Earth. Each level has new challenges and your ultimate goal is to build a communication network and synchronize, or match, the time on atomic clocks.\n\nAbout the mission\nOptical communication uses lasers to send data with a type of light that we can’t see, called infrared light. Since the 1950s, NASA missions have used a different type of waves, called radio waves, to send data to and from space. With current radio wave communications, it takes about nine weeks to send a complete map of Mars back to Earth. With lasers, it only takes about nine days!\nOptical communication is very useful for space missions because data can be sent faster and more securely. Optical communication systems are also smaller, weigh less, and use less power. A smaller size means more room for science instruments, and using less power means less of a drain of the spacecraft’s batteries. These are all very important details for NASA when designing and creating mission concepts. To learn more, visit Laser Communications | NASA.\n\n        \tarticle last updated July 28, 2022\n        \n\nIf you liked this, you may like:\n\nDSN Uplink-Downlink: A DSN Game\n\nSnap it! An Eclipse Photo Adventure\n\nExplore Mars: A Mars Rover Game\n\n\t    Games\t    \n\n\t    Crafts\t    \n\n\t    Activities\t    \n\n\t    Videos\t    \n\n\t    Glossary\t    \n\n\t    Mystery\t    \n\nAbout Us\nPrivacy Policy\nImage Use\nAccessibility\n\nContact NASA Space Place\n\n        Last Updated: May 6th, 2024    \n\n \n\nMore\nLess\n\nBy Subject\n\nSpace\nSun\nEarth\nSolar System\nPeople & Technology\nParents & Educators\n\nBy Type\n\nExplore\nDo\nPlay\n\nMore\nLess',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6196, 0.3624],
#         [0.6196, 1.0000, 0.3522],
#         [0.3624, 0.3522, 1.0000]])
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

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `validation`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.2845     |
| cosine_accuracy@3   | 0.4741     |
| cosine_accuracy@5   | 0.5431     |
| cosine_accuracy@10  | 0.6207     |
| cosine_precision@1  | 0.2845     |
| cosine_precision@3  | 0.2328     |
| cosine_precision@5  | 0.1776     |
| cosine_precision@10 | 0.1164     |
| cosine_recall@1     | 0.1134     |
| cosine_recall@3     | 0.2737     |
| cosine_recall@5     | 0.3382     |
| cosine_recall@10    | 0.4389     |
| **cosine_ndcg@10**  | **0.3513** |
| cosine_mrr@10       | 0.3962     |
| cosine_map@100      | 0.2961     |

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

* Size: 2,879 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                           |
  |:--------|:------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                               |
  | details | <ul><li>min: 12 tokens</li><li>mean: 70.99 tokens</li><li>max: 200 tokens</li></ul> | <ul><li>min: 27 tokens</li><li>mean: 308.02 tokens</li><li>max: 512 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>\|user\|: does being active affect how kids do in school? how?</code>                                                                                                                                                                                                                                                                                                                                                                                                                      | <code>Walk. Run. Dance. Play. What's your move? - Move Your Way \| health.gov Walk. Run. Dance. Play. What's your move? - Move Your Way \| health.gov<br>Get your kids moving<br><br>You know kids need physical activity to grow up strong and healthy.But did you know it can help them feel better right away?Better sleepBetter moodBetter gradesAnd when you move with your kids, you get the health benefits, too!<br><br>Help your kids move moreUse our interactive graphic to see how they can get the recommended 60 minutes a day.<br><br>Learn the basics<br><br>Find out how your kids can get enough physical activity. Read the fact sheet for parents [PDF - 1.9 MB].<br><br>Learn how sports can help your kids get active and build important life skills. Check out the sports fact sheet for parents [PDF - 1.7 MB].<br><br>Help children understand why physical activity is important. Share this fact sheet with your kids [PDF - 1.4 MB].<br><br>Watch our videos<br><br>Get moving as a family! See how you can get active together [YouTube – 1:55].<br><br>Stuck indoors? Lear...</code> |
  | <code>\|user\|: what are the different types of dialog nodes?<br>\|user\|: whatar are intents?<br>\|user\|: how is it created?<br>\|user\|: are those the only steps?<br>\|user\|: are dialogue skills necessary?<br>\|user\|: what is the difference with the dialog node?<br>\|user\|: which type can i create?<br>\|user\|: are there any advantages to adding a skill to my assistant?<br>\|user\|: search skill</code>                                                                      | <code><br>[Diagram of a more complex implementation that uses intent, entity, and dialog.](https://cloud.ibm.com/docs-content/v1/content/417db917d18067d9dcc4cf2612564a9a87c3ddc8/assistant/images/complex-impl.png)<br><br>As you add information, the skill uses this unique data to build a machine learning model that can recognize these and similar user inputs. Each time you add or change the training data, the training process is triggered to ensure that the underlying model stays up-to-date as your customer needs and the topics they want to discuss change.<br><br><br><br><br><br> Search skill ![Plus or higher plans only](https://cloud.ibm.com/docs-content/v1/content/417db917d18067d9dcc4cf2612564a9a87c3ddc8/assistant/images/plus.png) <br><br>When Watson Assistant doesn't have an explicit solution to a problem, it routes the user question to a search skill to find an answer from across your disparate sources of self-service content. The search skill interacts with the IBM Watson® Discovery service to extract this information fr...</code>                           |
  | <code>\|user\|: where do the arizona cardinals play this week<br>\|user\|: Do the Arizona Cardinals play outside the US?<br>\|user\|: Are the Arizona Cardinals and the Chicago Cardinals the same team?<br>\|user\|: How many teams are in the NFL?<br>\|user\|: How many teams are in the NFL playoffs?<br>\|user\|: Which team has won the most Super Bowls?<br>\|user\|: How many times have the New England Patriots played the super bowl?<br>\|user\|: Who is the Patriot's coach?</code> | <code>New England Patriots New England Patriots<br>Bill Belichick achieved his 200th career head coaching win ( regular season and playoffs ) on November 22 , 2012 , defeating the Jets 49 -- 19 ; it was his 163rd such win as Patriots coach . The Patriots defeated the Jets in Week seven of the 2015 season by a score of 30 -- 23 , to give them a 6 -- 0 record to date .</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
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

- `eval_strategy`: steps
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
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
- `num_train_epochs`: 3
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

### Training Logs
| Epoch | Step | validation_cosine_ndcg@10 |
|:-----:|:----:|:-------------------------:|
| 1.0   | 15   | 0.1692                    |
| 2.0   | 30   | 0.2353                    |
| 3.0   | 45   | 0.3513                    |


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