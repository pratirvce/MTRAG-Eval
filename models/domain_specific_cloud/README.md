---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:339
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: '|user|: Does Vulnerability Advisor scan encrypted imagery?

    |user|: Then what does Vulnerability Advisor scan? what does it actually do?

    |user|: I currently have Vulnerability Advisor version 3...

    |user|: how can I get this Vulnerability Advisor v4?

    |user|: what is difference between Vulnerability Advisor version 3 and version
    4?

    |user|: What if I want to create encrypted image since Vulnerability Advisor can
    not scan it?

    |user|: can you clarify IBM Cloud® Container Registry you are mentioning here?

    |user|: Container Registry Grant Access

    |user|: I do not know how to set up namespaces ...'
  sentences:
  - "\nPlans and provisioning \n\nIBM® Cloudant® for IBM Cloud® Standard is IBM Cloudant's\
    \ most feature-rich offering, receiving updates and new features first. Pricing\
    \ is based on provisioned throughput capacity that is allocated and data storage\
    \ that is used, making it suitable for any required load.\n\nThe free [Lite plan](https://cloud.ibm.com/docs/services/Cloudant?topic=Cloudant-ibm-cloud-publiclite-plan)\
    \ includes a fixed amount of throughput capacity and data for development and\
    \ evaluation purposes. The paid [Standard plan](https://cloud.ibm.com/docs/services/Cloudant?topic=Cloudant-ibm-cloud-publicstandard-plan)\
    \ offers configurable provisioned throughput capacity and data storage pricing\
    \ that scales as your application requirements change. An optional [Dedicated\
    \ Hardware plan](https://cloud.ibm.com/docs/services/Cloudant?topic=Cloudant-ibm-cloud-publicdedicated-hardware-plan)\
    \ is also available for an extra monthly fee to run one or more of your Standard\
    \ plan instances on a dedicated hardware environment. The dedicated hardware environment\
    \ is for your sole use. If a Dedicated Hardware plan instance is provisioned within\
    \ a US location, you can optionally select a [HIPAA](https://en.wikipedia.org/wiki/Health_Insurance_Portability_and_Accountability_Act)\
    \ -compliant configuration.\n\n\n\n Plans \n\nYou can select which plan to use\
    \ when you [provision your IBM Cloudant service instance](https://cloud.ibm.com/docs/services/Cloudant?topic=Cloudant-ibm-cloud-publicprovisioning-a-cloudant-nosql-db-instance-on-ibm-cloud).\
    \ The available plans include Lite and Standard. When you select a plan, Capacity\
    \ displays, and the Cost estimator shows the monthly charge for the selected plan.\
    \ By default, the [Lite plan](https://cloud.ibm.com/docs/services/Cloudant?topic=Cloudant-ibm-cloud-publiclite-plan)\
    \ is selected.\n\n\n\n Lite plan \n\nThe Lite plan is free, and is designed for\
    \ development and evaluation purposes."
  - "\nIBM Cloud Container Registry provides a multi-tenant private image registry\
    \ that you can use to store and share your container images with users in your\
    \ IBM Cloud account. Select the location for your namespace, and click Create.\
    \ [Learn more.](https://cloud.ibm.com/docs/Registry?topic=Registry-getting-started)\n\
    \n\n\n\n\n Step 5: Complete optional steps \n\nComplete any or all of the following\
    \ optional steps. If you don't complete these steps now, you can complete them\
    \ in the next tutorial.\n\n\n\n1. Create an [image signing key](https://cloud.ibm.com/docs/devsecops?topic=devsecops-devsecops-image-signing)\
    \ with the proper encoding to sign your application docker images.\n2. Create\
    \ an [IBM Cloud API key](https://cloud.ibm.com/iam/apikeys). Save the API key\
    \ value by either copying, downloading it or adding it to your vault. Alternatively,\
    \ you can create the API key during the template-guided setup process.\n3. Validate\
    \ that the [recommended IAM permissions](https://cloud.ibm.com/docs/devsecops?topic=devsecops-iam-permissions)\
    \ are assigned to corresponding integrations.\n4. [Install the IBM Cloud CLI](https://cloud.ibm.com/docs/cli?topic=cloud-cli-getting-started).\n\
    5. Create an [IBM Cloud Object Storage instance and bucket](https://cloud.ibm.com/docs/devsecops?topic=devsecops-cd-devsecops-cos-config).\
    \ [Learn more.](https://cloud.ibm.com/docs/cloud-object-storage?topic=cloud-object-storage-about-cloud-object-storage)\n\
    6."
  - "\nHowever, you can specify a larger volume size, different IOPS profile, and\
    \ different CRK if you prefer.\n\n\n\nYou can restore volumes from a manually\
    \ created snapshot or from a snapshot that was created by a backup policy. This\
    \ type of snapshot is called a backup. For more information, see [Restoring a\
    \ volume from a backup snapshot](https://cloud.ibm.com/docs/vpc?topic=vpc-baas-vpc-restore).\n\
    \nYou can restore a volume in a different region by using a cross-regional copy\
    \ of a snapshot. For more information, see [Cross-regional snapshots](https://cloud.ibm.com/docs/vpc?topic=vpc-snapshots-vpc-aboutsnapshots_vpc_crossregion_copy).\n\
    \nYou can also choose to restore a volume by using a fast restore snapshot clone.\
    \ For more information about fast restore, see the [FAQs](https://cloud.ibm.com/docs/vpc?topic=vpc-snapshots-vpc-faqs&interface=uifaq-snapshot-fr).\n\
    \nYou can restore volumes at various stages of the VPC lifecycle.\n\n\n\n* When\
    \ you provision a virtual server instance, you can specify a snapshot of a boot\
    \ or a snapshot of data volume. The restored boot volume is used to start the\
    \ new instance. Restored data volumes are automatically attached to the instance\
    \ as auxiliary storage.\n* When you want to add a new auxiliary storage to your\
    \ existing instance, you can restore a data volume from a nonbootable snapshot.\n\
    * When you create an unattached (stand-alone) Block Storage for VPC volume from\
    \ a snapshot, you can still attach the volume to an instance later.\n\n\n\n\n\n\
    \n\n Limitations of restoring a volume from a snapshot \n\nThe following limitations\
    \ apply when you restore a volume from a snapshot.\n\n\n\n* To restore a volume,\
    \ the snapshot must be in a stable state.\n* You can delete the new volume at\
    \ any time. However, you can't delete the snapshot from which the volume is restored\
    \ from unless the hydration is complete or the volume is deleted."
- source_sentence: '|user|: What are the steps to be taken to gather the relevant
    worker node data?

    |user|: How can i update a classic worker node?

    |user|: Major. menor update.

    |user|: parts of a tag.

    |user|: node data

    |user|: NodeSync

    |user|: Worker node'
  sentences:
  - "\n* VPC clusters: Worker nodes are provisioned in to an IBM Cloud account that\
    \ is owned by IBM to enable monitoring of malicious activities and apply security\
    \ updates. You can't access your worker nodes by using the VPC dashboard. However,\
    \ you can manage your worker nodes by using the IBM Cloud Kubernetes Service console,\
    \ CLI, or API. The virtual machines that make up your worker nodes are dedicated\
    \ to you and you are responsible to request timely updates so that your worker\
    \ node OS and IBM Cloud Kubernetes Service components apply the latest security\
    \ updates and patches.\n\n\n\nFor more information, see [Your responsibilities\
    \ by using IBM Cloud Kubernetes Service](https://cloud.ibm.com/docs/containers?topic=containers-responsibilities_iks).\n\
    \nUse the ibmcloud ks worker update[command](https://cloud.ibm.com/docs/containers?topic=containers-kubernetes-service-clics_worker_update)\
    \ regularly (such as monthly) to deploy updates and security patches to the operating\
    \ system and to update the Kubernetes version that your worker nodes run. When\
    \ updates are available, you are notified when you view information about the\
    \ master and worker nodes in the IBM Cloud console or CLI, such as with the ibmcloud\
    \ ks clusters ls or ibmcloud ks workers ls --cluster <cluster_name> commands.\
    \ Worker node updates are provided by IBM as a full worker node image that includes\
    \ the latest security patches. To apply the updates, the worker node must be reimaged\
    \ and reloaded with the new image. Keys for the root user are automatically rotated\
    \ when the worker node is reloaded.\n\n\n\n\n\n How does my worker node setup\
    \ look? \n\nThe following image shows the components that are set up for every\
    \ worker node to protect your worker node from malicious attacks.\n\nThe image\
    \ does not include components that ensure secure end-to-end communication to and\
    \ from the worker node. For more information, see [network security](https://cloud.ibm.com/docs/containers?topic=containers-securitynetwork).\n\
    \nZoom\n\n![Worker node setup in IBM Cloud Kubernetes Service excluding network\
    \ security.]"
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
  - "\n(https://cognitiveclass.ai/courses/docker-essentials)\n\n\n\n\n\n What is Red\
    \ Hat OpenShift? \n\nRed Hat OpenShift is a Kubernetes container platform that\
    \ provides a trusted environment to run enterprise workloads. It extends the Kubernetes\
    \ platform with built-in software to enhance app lifecycle development, operations,\
    \ and security. With Red Hat OpenShift, you can consistently deploy your workloads\
    \ across hybrid cloud providers and environments. For more information about the\
    \ differences between the community Kubernetes and Red Hat OpenShift cluster offerings,\
    \ see the [comparison table](https://cloud.ibm.com/docs/openshift?topic=openshift-cs_ovopenshift_kubernetes).\n\
    \n\n\n\n\n What compute host infrastructure does the service offer? \n\nYou can\
    \ create clusters on Classic or IBM Cloud® Virtual Private Cloud infrastructure.\
    \ You can also bring your own hosts by using Satellite.\n\nFor more information,\
    \ see [Supported infrastructure providers](https://cloud.ibm.com/docs/openshift?topic=openshift-infrastructure_providers).\n\
    \n\n\n\n\n Related resources \n\nReview how you can learn about Kubernetes concepts\
    \ and the terminology.\n\n\n\n* Familiarize yourself with the product by completing\
    \ the [Creating clusters tutorial](https://cloud.ibm.com/docs/openshift?topic=openshift-openshift_tutorial).\n\
    * Learn how Kubernetes and Red Hat OpenShift on IBM Cloud work together by completing\
    \ this [course](https://cognitiveclass.ai/courses/kubernetes-course)."
- source_sentence: '|user|: Does Vulnerability Advisor scan encrypted imagery?'
  sentences:
  - "\nTo find out more about the issues, click the link in the Security status column.\n\
    \nThe Vulnerability Advisor dashboard provides an overview and assessment of the\
    \ security for an image. If you want to find out more about the Vulnerability\
    \ Advisor dashboard, see [Reviewing a vulnerability report](https://cloud.ibm.com/docs/Registry?topic=Registry-va_index&interface=uiva_reviewing).\n\
    \nEncrypted images aren't scanned by Vulnerability Advisor.\n\n\n\n Data protection\
    \ \n\nTo scan images and containers in your account for security issues, Vulnerability\
    \ Advisor collects, stores, and processes the following information:\n\n\n\n*\
    \ Free-form fields, including IDs, descriptions, and image names (registry, namespace,\
    \ repository name, and image tag)\n* Metadata about the file modes and creation\
    \ timestamps of the configuration files\n* The content of system and application\
    \ configuration files in images and containers\n* Installed packages and libraries\
    \ (including their versions)\n\n\n\nDo not put personal information into any field\
    \ or location that Vulnerability Advisor processes, as identified in the preceding\
    \ list.\n\nScan results, aggregated at a data center level, are processed to produce\
    \ anonymized metrics to operate and improve the service. In version 3, a vulnerability\
    \ report (scan result) is generated when the image is pushed to the registry (and\
    \ is regenerated regularly thereafter). When Vulnerability Advisor is queried,\
    \ a scan result is retrieved that might be up to 5 days old. Scan results are\
    \ deleted 30 days after they are generated.\n\nIn version 4, the image is indexed\
    \ when it is first pushed to Container Registry registry, and that index report\
    \ is stored in the database. When Vulnerability Advisor is queried, the image\
    \ index report is retrieved, and a vulnerability assessment is produced. This\
    \ action happens dynamically every time Vulnerability Advisor is queried. Therefore,\
    \ no pregenerated scan result exists that requires deleting. However, the image\
    \ index report is deleted within 30 days of the deletion of the image from the\
    \ registry.\n\n\n\n\n\n\n\n Types of vulnerabilities \n\n\n\n Vulnerable packages\
    \ \n\nVulnerability Advisor checks for vulnerable packages in images that are\
    \ using supported operating systems and provides a link to any relevant security\
    \ notices about the vulnerability.\n\nPackages that contain known vulnerability\
    \ issues are displayed in the scan results."
  - "\nDeployment Journey Overview \n\nIBM Cloud® Virtual Private Cloud(VPC) allows\
    \ you to establish your own virtual private cloud by defining a virtual network\
    \ that is logically isolated from all other public cloud tenants. The underlying\
    \ software defined networking (SDN) and virtual network functions allows you to\
    \ quickly establish the network constructs and on-prem connectivity needed to\
    \ run your workload. The information contained within this document is meant to\
    \ serve as a technical guide for beginning with a new IBM Cloud Account and leading\
    \ towards a fully configured VPC network environment.\n\nWelcome to the Deployment\
    \ Journey for VPC on IBM Cloud! Use the sidebar on the left to navigate between\
    \ the journey points.\n\n\n\n Journey Map \n\n![Architecture](https://cloud.ibm.com/docs-content/v1/content/d50364af907081c3c38f99c11f35dd9cab4b2510/vpc-journey/images/overview/journey-map.png)\n\
    \n\n\n\n\n Assumptions \n\nThis deployment guide will be assuming the following\
    \ points. Please note that while your circumstance may not be exactly identical,\
    \ you will still benefit from the overall journey steps and concepts covered in\
    \ this guide.\n\n\n\n* You are already familar with the concepts introduced in\
    \ the \"Tour IBM Cloud\" videos available on the [Getting Started with IBM Cloud](https://cloud.ibm.com/cloud/get-started)\
    \ page.\n* Access groups will need to be defined so only certain users have the\
    \ ability to create and manage the VPC network settings (i.e. CIDR ranges, Subnet\
    \ ACL rules, etc.,).\n* You have a networking background and familar with concepts\
    \ such as IP Addressing, subnets, routing, etc.,\n* Focus will be on establishing\
    \ the underlying network connectivity to support VPC based workloads.\n\n\n\n\
    * Note: A separate deployment guide will cover the compute resources which runs\
    \ within the VPC like IBM Kubernetes Services (IKS), Red Hat OpenShift, IBM Code\
    \ Engine, and VPC Virtual Server Instances (VSIs)."
  - "\nA measure of a test's accuracy that considers both precision and recall to\
    \ compute the score. The F1 score can be interpreted as a weighted average of\
    \ the precision and recall values. An F1 score reaches its best value at 1 and\
    \ worst value at 0.\n\n\n\n\n\n false negative \n\nAn answer or annotation that\
    \ is correct, but was predicted to be incorrect.\n\n\n\n\n\n false positive \n\
    \nAn answer or annotation that is incorrect, but was predicted to be correct.\n\
    \n\n\n\n\n feature \n\nA data member or attribute of a type.\n\n\n\n\n\n feature\
    \ code \n\nA code that is applied to free accounts to unlock extra product resources\
    \ and capabilities.\n\n\n\n\n\n Federal Risk and Authorization Management Program\
    \ (FedRAMP) \n\nA United States government program that provides a standardized,\
    \ risk-based approach for the adoption and use of cloud services by the US federal\
    \ government. FedRAMP empowers agencies to use modern cloud technologies with\
    \ an emphasis on security and protection of federal information, and helps accelerate\
    \ the adoption of secure cloud solutions.\n\n\n\n\n\n federate \n\nTo merge two\
    \ or more entities. For example, a company's registered domain could be federated\
    \ with an IBMid.\n\n\n\n\n\n FedRAMP \n\nSee [Federal Risk and Authorization Management\
    \ Program](https://cloud.ibm.com/docs/overview?topic=overview-glossaryx10109081).\n\
    \n\n\n\n\n feed \n\nA piece of code that configures an external event source to\
    \ fire trigger events. See also [action](https://cloud.ibm.com/docs/overview?topic=overview-glossaryx2012974).\n\
    \n\n\n\n\n file share \n\nIn the IBM Cloud environment, a persistent storage system\
    \ where users store and share files. In IBM Cloud Kubernetes Service, users can\
    \ mount Docker volumes on file shares.\n\n\n\n\n\n fire \n\nTo activate a trigger.\n\
    \n\n\n\n\n Fleiss Kappa score \n\nA measure of how consistently the same annotation\
    \ was applied by multiple human annotators across overlapping documents. The Fleiss\
    \ Kappa score reaches its best value at 1 and worst value at 0.\n\n\n\n\n\n floating\
    \ IP address \n\nA public, routable IP address that makes use of 1-to-1 network\
    \ address translation (NAT) so that a server can communicate with the public internet\
    \ and private subnet within a cloud environment."
- source_sentence: "|user|: What are the billing plans available and differences between\
    \ each of them?\n|user|: how can I estimate my costs?\n|user|: are there different\
    \ types of charges I am charged? \n|user|: can you tell me the difference among\
    \ Fixed, Metered, Tiered, and Reserved?\n|user|: I want to update my pricing plan\
    \ and view my usage \n|user|: am I charged for those support costs as well?\n\
    |user|: what is the process of managing my payments?"
  sentences:
  - "\nIf you're a new Pay-As-You-Go account owner that is located in the US and you\
    \ are paying with a credit card, can you add multiple cards to the account, replace\
    \ your default card with a saved one, or edit the details of a card. You manage\
    \ your credit card from the [Payments](https://cloud.ibm.com/billing/payments)\
    \ page in the IBM Cloud console.\n\nComplete the following steps to add a new\
    \ payment method to the account:\n\n\n\n1. Click Add payment method.\n2. Enter\
    \ the card details, and click Save. Updates to your card details are reflected\
    \ immediately.\n\n\n\nYou can’t enter a PO Box as the billing address.\n\nWhen\
    \ you add a new credit card, it becomes the default credit card. Recurring payments\
    \ are charged to the default payment method.\n\nComplete the following steps to\
    \ edit your active payment method:\n\n\n\n1. Click the Actions icon ![Actions\
    \ icon](https://cloud.ibm.com/docs-content/v1/content/9713d864488177dc6b273b53b7f2383a81f10bc1/icons/action-menu-icon.svg)\
    \ > Edit menu.\n2. To edit the billing address, click Edit and update the billing\
    \ address.\n3. To edit the card details, click Edit and update the card number\
    \ or expiration date.\n\n\n\nYou can only have one address that's associated with\
    \ your payment methods. All credit cards in the account will be updated to the\
    \ same address.\n\nComplete the following steps to set a new default payment method:\n\
    \n\n\n1. Click the Actions icon ![Actions icon](https://cloud.ibm.com/docs-content/v1/content/9713d864488177dc6b273b53b7f2383a81f10bc1/icons/action-menu-icon.svg)\
    \ > Set as default.\n2. Confirm that you want to make this payment method the\
    \ default. The default payment method is charged for recurring payments\n\n\n\n\
    \n\n\n\n Managing payment methods for all other accounts"
  - "\n(Optional) [Create an acurl alias](https://cloud.ibm.com/docs/Cloudant?topic=Cloudant-working-with-curlencode-user-name-and-password).\n\
    \n\n\nIf you decide not to set up acurl, use the following URL with curl instead\
    \ of the one provided in the exercises, curl \"https://$USERNAME:$PASSWORD@$ACCOUNT.cloudant.com/databasedemo\"\
    .\n\nThe acurl alias is more secure. It prevents someone from reading your password\
    \ over your shoulder as you type. It also makes sure that your password isn’t\
    \ sent in plain text over the network by enforcing HTTPS.\n\nNow, we're ready\
    \ to learn how to run queries against the database you created in step two of\
    \ [Before you begin](https://cloud.ibm.com/docs/Cloudant?topic=Cloudant-creating-an-ibm-cloudant-querybefore-you-begin-qt).\n\
    \n\n\n\n\n Step 1: Creating an index \n\nIBM Cloudant Query uses Mongo-style query\
    \ syntax to search for documents by using logical operators. IBM Cloudant Query\
    \ is a combination of a view and a search index.\n\nWhen you use IBM Cloudant\
    \ Query, the query planner looks at the selector (your query) to determine the\
    \ correct index to choose from. In memory, you filter out the documents by the\
    \ selector, which is why, even without an index, you can still query with various\
    \ fields.\n\nIf no available defined index matches the specified query, then IBM\
    \ Cloudant uses the _all_docs index, which looks up documents by ID. In the worst\
    \ case scenario, it returns all the documents by ID (full table scan). Full table\
    \ scans are expensive to process. It is recommended that you create an index.\n\
    \nTo create an index, follow these steps:\n\n\n\n1. Copy the following sample\
    \ JSON data into a file named query-demo-index.json:\n\n{\n\"index\": {\n\"fields\"\
    : [\n\"descriptionField\",\n\"temperatureField\"\n],\n\"partial_filter_selector\"\
    : {\n\"descriptionField\": {"
  - "\n} else {\nvar result = reply[\"result\"]\nprint(\"Got result (result)\")\n\
    }\n})\n} catch {\nprint(\"Error (error)\")\n}\n\nBy default, the SDK returns only\
    \ the activation ID and any result that is produced by the invoked action. To\
    \ get metadata of the entire response object, which includes the HTTP response\
    \ status code, use the following setting:\n\nwhisk.verboseReplies = true\n\n\n\
    \n\n\n Configuring the mobile SDK \n\nYou can configure the SDK to work with different\
    \ installations of Cloud Functions by using the baseURL parameter. For instance:\n\
    \nwhisk.baseURL = \"http://localhost:8080\"\n\nIn this example, you use an installation\
    \ that is running at http://localhost:8080. If you do not specify the baseURL,\
    \ the mobile SDK uses the instance that is running at [https://us-south.functions.cloud.ibm.com](https://us-south.functions.cloud.ibm.com).\n\
    \nYou can pass in a custom NSURLSession in case you require special network handling.\
    \ For example, you might have your own Cloud Functions installation that uses\
    \ self-signed certificates:\n\n// create a network delegate that trusts everything\n\
    class NetworkUtilsDelegate: NSObject, NSURLSessionDelegate {\nfunc URLSession(session:\
    \ NSURLSession, didReceiveChallenge challenge: NSURLAuthenticationChallenge, completionHandler:\
    \ (NSURLSessionAuthChallengeDisposition, NSURLCredential?) -> Void) {\ncompletionHandler(NSURLSessionAuthChallengeDisposition.UseCredential,\
    \ NSURLCredential(forTrust: challenge.protectionSpace.serverTrust!))\n}\n}\n//\
    \ create an NSURLSession that uses the trusting delegate"
- source_sentence: "|user|: What are the billing plans available and differences between\
    \ each of them?\n|user|: how can I estimate my costs?\n|user|: are there different\
    \ types of charges I am charged? \n|user|: can you tell me the difference among\
    \ Fixed, Metered, Tiered, and Reserved?\n|user|: I want to update my pricing plan\
    \ and view my usage \n|user|: am I charged for those support costs as well?\n\
    |user|: what is the process of managing my payments?\n|user|: view my invoices\
    \ and status\n|user|: Why do you think I can not apply a subscription code?\n\
    |user|: what about why can't I apply a feature code?\n|user|: I also tried to\
    \ update my credit card, but it keeps showing errors\n|user|: it is showing \"\
    Could not place order. Problem authorizing the credit card. We are unable to process\
    \ your request: Transaction Rejected\""
  sentences:
  - "\nProblem authorizing the credit card. We're afraid this transaction has been\
    \ rejected. Inactive card or card not authorized for card-not-present transactions.\n\
    \n Why it’s happening \n\nSome credit card issuers don't allow transactions when\
    \ it is being initiated by using a credit card on file. You might see this error\
    \ message if your credit card has been deactivated.\n\n How to fix it \n\nContact\
    \ your credit card issuer for more information.\n\n\n\n\n\n Why is my credit card\
    \ not authorized to place an order? \n\n What’s happening \n\nYou tried to place\
    \ an order in the IBM Cloud console [Payments page](https://cloud.ibm.com/billing/payments),\
    \ but you get the following error message:\n\n> Could not place order. Problem\
    \ authorizing the credit card.\n\n Why it’s happening \n\nYour credit card issuer\
    \ has declined the transaction.\n\n How to fix it \n\nContact your credit card\
    \ issuer for more information.\n\n\n\n\n\n Why is IBM Cloud unable to process\
    \ my order request? \n\n What’s happening \n\nYou tried to place an order in the\
    \ IBM Cloud console [Payments page](https://cloud.ibm.com/billing/payments), but\
    \ you get the following error message:\n\n> Could not place order. Problem authorizing\
    \ the credit card. We are unable to process your request: Transaction Rejected.\
    \ Please contact Cloud Trust Enablement at [verify@us.ibm.com](mailto:verify@us.ibm.com).\n\
    \n Why it’s happening \n\nIBM Cloud® was unable to process your transaction.\n\
    \n How to fix it \n\nContact the Cloud Trust and Enablement team by email at [verify@us.ibm.com](mailto:verify@us.ibm.com).\n\
    \n\n\n\n\n Why was my change request rejected? \n\n What’s happening \n\nYou tried\
    \ to place an order in the IBM Cloud console [Payments page](https://cloud.ibm.com/billing/payments),\
    \ but you get one of the following error messages:\n\n> Failed to complete the\
    \ Change Request process due to the following error: We're afraid this transaction\
    \ has been rejected. Invalid account number.\n\nor"
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
    \ See [Analyzing low F1 scores](https://cloud.ibm.com/docs/watson-knowledge-studio-data?topic=watson-knowledge-studio-data-evaluate-mlevaluate-mllowf1).\n\
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
    \ scores](https://cloud.ibm.com/docs/watson-knowledge-studio-data?topic=watson-knowledge-studio-data-evaluate-mlevaluate-mllowp).\n\
    * Recall\n\nA measurement that specifies how many mentions that should have been\
    \ annotated by a given label were actually annotated with that label - the right\
    \ mentions being those that human annotators identified in the same documents."
  - "\nDownload the [Hive-compatible client](https://us.sql-query.cloud.ibm.com/download/catalog/hive-metastore-standalone-client-3.1.2-sqlquery-1.0.13.jar)\
    \ and place it in a directory of your Apache Spark cluster that is not on the\
    \ classpath. This step is necessary, as the client is loaded into an isolated\
    \ classloader to avoid version conflicts. Note that in the examples the files\
    \ are placed in /tmp/dataengine, when you use a different folder, adjust the example\
    \ accordingly.\n\nThe client differs from the Hive version 3.1.2 release by more\
    \ enhancements that add support for TLS and authentication through IBM Cloud®\
    \ Identity and Access Management. For user, specify the CRN and for password a\
    \ valid API key with access to your Data Engine. Find the endpoint to use in the\
    \ following table.\n\n\n\nTable 1. Region endpoints\n\n Region Endpoint \n\n us-south\
    \ thrift://catalog.us.dataengine.cloud.ibm.com:9083 \n eu-de thrift://catalog.eu-de.dataengine.cloud.ibm.com:9083\
    \ \n\n\n\n\n\n\n\n Convenience libraries to configure Spark \n\nWhile the Data\
    \ Engine catalog is compatible with the Hive metastore and can be used as any\
    \ other external Hive metastore server, an SDK is provided to minimize the steps\
    \ that are needed to configure Apache Spark. The SDK simplifies connecting to\
    \ the Hive metastore and IBM Cloud Object Storage buckets referenced by tables\
    \ or views.\n\nIn case of using Python download both, the Scala and the Python\
    \ SDK, and place them in a folder that is in the classpath of your Apache Spark\
    \ cluster. When using Scala, the Scala SDK is enough.\n\n\n\n* [spark-dataengine-scala](https://us.sql-query.cloud.ibm.com/download/catalog/dataengine-spark-integration-1.4.51.jar)"
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
- name: SentenceTransformer
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: cloud validation
      type: cloud_validation
    metrics:
    - type: cosine_accuracy@1
      value: 0.35714285714285715
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.4642857142857143
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.5714285714285714
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.6785714285714286
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.35714285714285715
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.22619047619047614
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.17857142857142858
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.11071428571428574
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.15595238095238093
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.27440476190476193
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.37083333333333324
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.4660714285714285
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.38062173375128455
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.44264455782312934
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.327667042134629
      name: Cosine Map@100
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
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
    '|user|: What are the billing plans available and differences between each of them?\n|user|: how can I estimate my costs?\n|user|: are there different types of charges I am charged? \n|user|: can you tell me the difference among Fixed, Metered, Tiered, and Reserved?\n|user|: I want to update my pricing plan and view my usage \n|user|: am I charged for those support costs as well?\n|user|: what is the process of managing my payments?\n|user|: view my invoices and status\n|user|: Why do you think I can not apply a subscription code?\n|user|: what about why can\'t I apply a feature code?\n|user|: I also tried to update my credit card, but it keeps showing errors\n|user|: it is showing "Could not place order. Problem authorizing the credit card. We are unable to process your request: Transaction Rejected"',
    "\nProblem authorizing the credit card. We're afraid this transaction has been rejected. Inactive card or card not authorized for card-not-present transactions.\n\n Why it’s happening \n\nSome credit card issuers don't allow transactions when it is being initiated by using a credit card on file. You might see this error message if your credit card has been deactivated.\n\n How to fix it \n\nContact your credit card issuer for more information.\n\n\n\n\n\n Why is my credit card not authorized to place an order? \n\n What’s happening \n\nYou tried to place an order in the IBM Cloud console [Payments page](https://cloud.ibm.com/billing/payments), but you get the following error message:\n\n> Could not place order. Problem authorizing the credit card.\n\n Why it’s happening \n\nYour credit card issuer has declined the transaction.\n\n How to fix it \n\nContact your credit card issuer for more information.\n\n\n\n\n\n Why is IBM Cloud unable to process my order request? \n\n What’s happening \n\nYou tried to place an order in the IBM Cloud console [Payments page](https://cloud.ibm.com/billing/payments), but you get the following error message:\n\n> Could not place order. Problem authorizing the credit card. We are unable to process your request: Transaction Rejected. Please contact Cloud Trust Enablement at [verify@us.ibm.com](mailto:verify@us.ibm.com).\n\n Why it’s happening \n\nIBM Cloud® was unable to process your transaction.\n\n How to fix it \n\nContact the Cloud Trust and Enablement team by email at [verify@us.ibm.com](mailto:verify@us.ibm.com).\n\n\n\n\n\n Why was my change request rejected? \n\n What’s happening \n\nYou tried to place an order in the IBM Cloud console [Payments page](https://cloud.ibm.com/billing/payments), but you get one of the following error messages:\n\n> Failed to complete the Change Request process due to the following error: We're afraid this transaction has been rejected. Invalid account number.\n\nor",
    "\nAnalyzing machine learning model performance \n\nReview the annotations that were added by the trained model to determine whether any adjustments must be made to the model to improve its ability to find valid entity mentions, relation mentions, and coreferences in the documents.\n\n\n\n About this task \n\nYou can analyze performance by viewing a summary of statistics for entity types, relation types, and coreferenced mentions. You can also analyze statistics that are presented in a confusion matrix. The confusion matrix helps you compare the annotations added by the machine learning model to the annotations in ground truth.\n\nThe model statistics provide the following metrics:\n\n\n\n* F1 score\n\nA measurement that considers both precision and recall to compute the score. The F1 score can be interpreted as a weighted average of the precision and recall values, where an F1 score reaches its best value at 1 and worst value at 0. See [Analyzing low F1 scores](https://cloud.ibm.com/docs/watson-knowledge-studio-data?topic=watson-knowledge-studio-data-evaluate-mlevaluate-mllowf1).\n* Precision\n\nA measurement that specifies what fraction of the machine learning model's output was accurate when compared to the human annotator output. Precision is determined by the number of correctly labeled annotations divided by the total number of annotations added by the machine learning model. A precision score of 1.0 for entity type A means that every mention that was labeled as entity type A does indeed belong to that classification. A low precision score helps you identify places where the machine learning model created incorrect annotations. The score says nothing about how many other mentions that were labeled as entity type A by the human annotator were missed by the machine learning model; the recall score reflects that information. See [Analyzing low precision scores](https://cloud.ibm.com/docs/watson-knowledge-studio-data?topic=watson-knowledge-studio-data-evaluate-mlevaluate-mllowp).\n* Recall\n\nA measurement that specifies how many mentions that should have been annotated by a given label were actually annotated with that label - the right mentions being those that human annotators identified in the same documents.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7817, 0.1297],
#         [0.7817, 1.0000, 0.1567],
#         [0.1297, 0.1567, 1.0000]])
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

* Dataset: `cloud_validation`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.3571     |
| cosine_accuracy@3   | 0.4643     |
| cosine_accuracy@5   | 0.5714     |
| cosine_accuracy@10  | 0.6786     |
| cosine_precision@1  | 0.3571     |
| cosine_precision@3  | 0.2262     |
| cosine_precision@5  | 0.1786     |
| cosine_precision@10 | 0.1107     |
| cosine_recall@1     | 0.156      |
| cosine_recall@3     | 0.2744     |
| cosine_recall@5     | 0.3708     |
| cosine_recall@10    | 0.4661     |
| **cosine_ndcg@10**  | **0.3806** |
| cosine_mrr@10       | 0.4426     |
| cosine_map@100      | 0.3277     |

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

* Size: 339 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 339 samples:
  |         | sentence_0                                                                          | sentence_1                                                                           |
  |:--------|:------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                               |
  | details | <ul><li>min: 12 tokens</li><li>mean: 69.74 tokens</li><li>max: 200 tokens</li></ul> | <ul><li>min: 84 tokens</li><li>mean: 427.19 tokens</li><li>max: 502 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>\|user\|: does IBM offer document databases?<br>\|user\|: So it can store any random JSON object or I need to specify fields in advances?<br>\|user\|: What if I want to store an image or pdf with a document? <br>\|user\|: Is there a limit on the file size? <br>\|user\|: Not documents, I was asking for attachments?<br>\|user\|: Ok. How do I get started? I just look through cloud catelog for IBM Cloudant? <br>\|user\|: What are difference between lite and standard plan?<br>\|user\|: So does that mean if I am just exploring things for now, I should stick to to Lite plan? </code> | <code><br>If you want to store more than one GB of data, or be able to scale provisioned throughput capacity, move to the [Standard plan](https://cloud.ibm.com/docs/services/Cloudant?topic=Cloudant-ibm-cloud-publicstandard-plan).<br><br>You're limited to one IBM Cloudant Lite plan instance per IBM Cloud account. If you already have one Lite plan instance, you can't create a second Lite plan instance, or change a Standard plan instance to a Lite plan. If you try, you see the following message. You can have only one instance of a Lite plan per service. To create a new instance, either delete your existing Lite plan instance or select a paid plan.<br><br><br><br><br><br> Standard plan <br><br>The IBM Cloudant Standard plan is available to all paid IBM Cloud® accounts, either as pay-as-you-go or subscription, and scales to meet the needs of your application. The Standard plan is priced based on two factors: the provisioned throughput capacity that is allocated, and the amount of data that is stored in the instance.<br><br>Pricing is ...</code>                               |
  | <code>\|user\|: Should we switch to a different method for installing the mobile SDK, considering the current approach's limitations?  <br>\|user\|: I meant the issues that may arise when installing it with CocoaPods. <br>\|user\|: No, there is a warning that comes up when installing it from CocoaPods.<br>\|user\|: Can I just not use CocoaPods?</code>                                                                                                                                                                                                                                            | <code><br>Mobile SDK <br><br>IBM Cloud® Functions provides a mobile SDK for iOS and watchOS devices that enables mobile apps to fire remote triggers and invoke remote actions. A version for Android is not available, so Android developers can use the OpenWhisk REST API directly. The mobile SDK is written in Swift 4 and supports iOS 11 and later releases. You can build the mobile SDK by using Xcode 9.<br><br>The mobile SDK is not supported for IAM-based namespaces. Use a Cloud Foundry-based namespace instead.<br><br><br><br> Add the SDK to your app <br><br>You can install the mobile SDK by using CocoaPods, Carthage, or from the source directory.<br><br><br><br> Install mobile SDK with CocoaPods <br><br>The IBM Cloud® Functions SDK for mobile is available for public distribution through CocoaPods. Assuming CocoaPods is installed, put the following lines into a file called Podfile inside the starter app project directory.<br><br>install! 'cocoapods', :deterministic_uuids => false<br>use_frameworks!<br><br>target 'MyApp' do<br>pod 'OpenWhisk', :git => 'https://git...</code> |
  | <code>\|user\|: What are the different types of dialog nodes?<br>\|user\|: Whatar are intents?<br>\|user\|: How is it created?<br>\|user\|: Are those the only steps?<br>\|user\|: Are dialogue skills necessary?<br>\|user\|: What is the difference with the dialog node?<br>\|user\|: Which type can I create?</code>                                                                                                                                                                                                                                                                                     | <code><br>Adding a skill to your assistant <br><br>Customize your assistant by adding to it the skills it needs to satisfy your customers' goals.<br><br>You can create the following types of skills:<br><br><br><br>* Dialog skill: Uses Watson natural language processing and machine learning technologies to understand user questions and requests, and respond to them with answers that are authored by you.<br>* Search skill: For a given user query, uses the IBM Watson® Discovery service to search a data source of your self-service content and return an answer.<br><br><br><br>Typically, you create a skill of each type first. Then, as you build a dialog for the dialog skill, you decide when to initiate the search skill. For some questions or requests, a hardcoded or programmatically-derived response (that is defined in the dialog skill) is sufficient. For others, you might want to provide a more robust response by returning a full passage of related information (that is extracted from an external data source by using the search skill).<br><br><br>...</code>                   |
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
- `num_train_epochs`: 7
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
- `num_train_epochs`: 7
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
| Epoch | Step | cloud_validation_cosine_ndcg@10 |
|:-----:|:----:|:-------------------------------:|
| 1.0   | 11   | 0.2070                          |
| 2.0   | 22   | 0.2343                          |
| 3.0   | 33   | 0.3225                          |
| 4.0   | 44   | 0.3527                          |
| 5.0   | 55   | 0.3629                          |
| 6.0   | 66   | 0.3755                          |
| 7.0   | 77   | 0.3806                          |


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