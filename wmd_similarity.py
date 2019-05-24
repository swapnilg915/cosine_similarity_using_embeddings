import json, traceback
import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.similarities import WmdSimilarity


class SemanticSimilarity(object):

	def __init__(self):
		self.stop_words = stopwords.words('english')
		self.w2v_path = '/home/swapnil/Downloads/GoogleNews-vectors-negative300.bin.gz'
		self.loadW2V()
		self.lemmatizer = WordNetLemmatizer()
	

	def cleanText(self, doc):

		doc = doc.replace("<br>","")
		doc = doc.replace(":","")
		doc = doc.replace("</br>","")
		doc = doc.replace("<b>","")
		doc = doc.replace("</b>","")
		doc = doc.replace("?","")
		doc = doc.replace("/","")
		doc = doc.replace("\\","")
		doc = doc.replace("-","")
		doc = doc.replace("\/","")
		doc = doc.replace("\n","")
		doc = doc.replace(")","")
		doc = doc.replace("(","")
		# doc = doc.replace(".","")
		doc = doc.lower().strip()
		return doc


	def word_token(self, tokens, lemma=False):
		tokens = str(tokens)
		# tokens = tokens.encode("utf-8")
		# tokens = tokens.encode("ascii",'ignore')
		tokens = re.sub(r"([\w].)([\~\!\@\#\$\%\^\&\*\(\)\-\+\[\]\{\}\/\"\'\:\;])([\s\w].)", "\\1 \\2 \\3", tokens)
		tokens = re.sub(r"\s+", " ", tokens)
		if lemma:
			return [self.lemmatizer.lemmatize(token) for token in word_tokenize(tokens.lower()) if token not in self.stop_words and token.isalpha()]
		else:
			return [token for token in word_tokenize(tokens.lower()) if token not in self.stop_words and token.isalpha()]

	def prepareData(self, resume):
		self.documents = [resume]
		wmd_corpus = list(map(lambda sent : self.word_token(sent), self.documents))
		print("\n wmd_corpus --- ", wmd_corpus)
		# wmd_corpus = [[tpl[0] for tpl in pos_tag(wmd_corpus[0]) if tpl[1] in ['NN', 'NNS','NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] ]]
		wmd_corpus = [[tpl[0] for tpl in pos_tag(wmd_corpus[0]) if tpl[1] in ['NN','VB'] ]]
		print("\n noun verb word corpus --- ",wmd_corpus)
		return wmd_corpus

	def loadW2V(self):
		if not os.path.exists(self.w2v_path):
			raise ValueError("SKIP: You need to download the google news model")
		self.w2v_model = KeyedVectors.load_word2vec_format(self.w2v_path, limit=500000, binary=True)

	def getWmdSimilarity(self, jd, resume):

		jd = "We are looking W2 Consultant only. Role Full stack UI Developer Location Mountain View CA Duration 3 Months Payroll team is working to enhance the customer setup and the ability to update employees payroll information to provide cleaner experience for the customers eliminating the need to call customer service. Responsibilities Designs codes tests debug and documents software and enhance existing components to ensure that software meets business needs. Contribute to the design and architecture to enable secure scalable and maintainable software. Apply technical expertise to resolve challenging programming and design problems. Front end UI design and development using JavaScript frameworks and HTML CSS and other Web technologies. Accurately estimate engineering work effort for engineering team. Build high quality code following unit testing and test driven development. Work cross functionally with various extended teams product management designers QA customer support and other business drivers to deliver product features and to address critical customer issues. Skills and Qualifications BS MS in Computer Science or equivalent work experience. 5 years of work experience developing scalable customer facing web and software applications. Strong experience leveraging technologies such as Java J2EE JavaScript HTML5 jQuery and related tools and web frameworks. 3 years of professional experience working with backbone.js or similar JavaScript frameworks is required. Experience in JSP JSF Struts Experience with XML JSON and developing REST services. Good understanding of SQL relational database preferably SQL Server Experienced in Agile software development and Scrum lifecycle practices."

		resume = " OZAN MANAV Istanbul Turkey www.ozanmanav.com +90 551 860 2015 Tranings Volunteer works - Microsoft Student Partner - Google Scholarship - Udemy Pluralsight Javascript Learning Path EDUCATION BACHELOR DEGREE ESKISEHIR OSMANGAZI UNIVERSITY BS in Computer engineering Sep 2012 June 2017 Responsibilities Works - I developed React Native and Native Mobile Applications. - QRcode Supported Membership System Android App Google Play- Elma Cafe Plus - IOT device management app based on logical values Google Play- Rim Control 4 AYTIM GROUP - TURKEY Aytim is the gaming textile company in Turkey providing services in all business lines. JUNIOR SOFTWARE ENGINEER Feb 2014 - Dec 2016 Responsibilities Works - I provided methodologies for object-oriented software development and efficient database design. - We have developed payment systems infrastructure together with the team. - I ve experienced UI testing frameworks like Selenium - I gave trainings to my team about Clean Code and TDD. From Robert C.Martin Books - I ve developed myself for secure software development. BDDK PCI Standards ARENA COMPUTER INC. or Payment Systems - TURKEY Arena is the leading provider of technology products and related supply chain management services in Turkey. Arena is characterised by its high level of innovation professional management and development strategies. Dec 2016 Jan 2018 SOFTWARE ENGINEER Im a Software Engineer familiar with a wide range of programming utilities and languages. Knowledgeable of backend and frontend development requirements. Able to handle any part of the process with ease. Collaborative team player with excellent technical abilities offering 4 years of related experience. JOB EXPERIENCESSOFTWARE ENGINEER Jan 2018 - Current ATP Ata Technology Platform - TURKEY ATP a leader in finance technologies addresses the needs of brokerage firms portfolio managers and insurance companies with comprehensive solutions and services. Its platforms handle a significant portion of the Istanbul Stock Exchanges trading volume. Responsibilities Works - Im supporting frontend mobile and web development and improvement process - Weve developed a dashboard with ReactJS and continuing maintenance with my team friends. - I ve developed react native screens in some parts of Native mobile projects. - In addition I developed mobile applications with React Native for Shiftdelete.net one of Turkey s largest tech news sites. ozan.manavv@gmail.com GraphQL AWS Docker TDD or BDD Agile or Scrum RESTFul APIs Node Webpack Git HTML5 CSS3 ES6 SOFTWARE ENGINEER Javascript React or React Native Redux CORE SKILLS PROFESSIONAL SUMMARY "

		resume = " Microsoft Word - Shalini Channappa.docx Shalini Channappa Front End Web Developer talentenvoy.shalini@gmail.com Hard-working web developer with a flair for creating elegant solutions in the least amount of time. Passionate about building responsive websites mobile apps and interactive features that drive business growth and improve UX. Experience Front End Web Developer Cisco 07 or 2018 - present Develop and test new components for the Digital partner advisor DPA project using Cisco UI Angular. Experience in developing single page applications using Angular. Improvise existing components and usability of various areas of the application working closely with a Product manager. Work in an Agile Scrum methodology on fast-moving projects. Extensive experience in UI web applications using HTML5 CSS3 Javascript XML jQuery AJAX JSON Angular and integrating Restful API s. Worked on eliminating bootstrap one of the two UI libraries of the application in order to avoid bloat overwriting and conflicts. Also handled the aftermath of the breakdown of layout and components and stabilized the application with release readiness in one sprint. Upgraded DPA to the current version of Cisco UI which was six versions behind and 90 of the library being overridden by custom definitions. Freelance Web Developer 08 or 2016 - 05 or 2018 Clients Turbo Tax Gabes Rentelo GPA Saver Translated design teams UX wireframes and mockups into responsive interactive features using HTML CSS and JavaScript. Worked with agile team to migrate legacy company website to a Wordpress site. Redesign of Gabes Android mobile app which increased downloads by 18 in less than 6 months. Increased email signups 12 by creating new UI for website landing page in React. Created highly detailed and annotated architectural wireframes. Successfully submitted MVPS. Actively participated in slack channels daily standups UI or UX design process code reviews responsive design managing project using Github s project Kanban board interface documentation testing and the final product launch. Manager Risk Investigations Amazon.com 09 or 2012 - 08 or 2016 Created grease monkey scripts to improve manual investigation efficiency by 115 . Created a script to review investigation steps dynamically and enable mistake proofing to improve investigation quality and reduce decision defect. Conducted a six sigma yellow belt Kaizen event with business operations analytics and software development team to determine and build machine learning model and variable to reduce incoming volume by 45 and saved 7.5 MM. Created dashboard for Amazon.in category management team using ETL jobs. Web Developer Intern Hindustan Aeronautics Limited 01 or 2011 - 05 or 2011 Designed UX wireframes and mockups and translated into interactive features using HTML CSS and JavaScript. Involved in writing stored procedures queries triggers and views. Wrote SQL queries to interact with SQL Server database. Web Developer Intern E Surveying Softtech 08 or 2010 - 12 or 2010 Handled search engine optimization SEO for the company s website resulting in which the website managed to top the Google search in survey related software. Performed Manual Testing on newly launched software technical content writing for upcoming software releases and web content development. EDUCATION Texas A M University 08 or 2016 - 05 or 2018 Master Of Science Computer Science GPA 3.91 SKILLS HTML CSS SQL JavaScript UI or UX Design Angular React Native AWARDS - Above and beyond awards in Q1 and Q3 of 2015 from Amazon.com - Received 6 employee of the month awards from Amazon.com - Awarded as the best Quality auditor during my tenure as quality auditor - Best new hire trainee from a batch of twelve in Amazon.com - Recipient of Grow with Google Developer Challenge Scholarship "

		resume = " Microsoft Word - Raviteja Kondubhatla.docx Ravteja Kondubhatla Data Scientist talentenvoy.raviteja@gmail.com Summary With my 5 years of experience in coding with analytical programming using Python SQL and Hadoop Id like to plan design and implement database solutions and work cross-functionally to customize client needs. My passion is to develop web application back end components and offer support to the front-end developers. Experience Data Scientist Cuna Mutual Group Wisconsin USA Oct 2018 - Present Implemented discretization and binning data wrangling cleaning transforming merging and reshaping data frames using python libraries like Numpy Scikit Matplotlib and Pandas Developed a propensity score generator for targeting the prospective Credit Union members using Machine Learning algorithms using Python Data Analyst Python Development Samsung California USA May 2018-Jul 2018 Automated batch test evaluation that allows a smooth flow of data from distributed data systems to the local machines and involved in Unit testing and Integration testing of the code Created a text normalizer using NLP for Bixby modules and created a workflow using technologies such as GIT Gained experience in working with various Python Integrated Development Environments like IDLE PyCharm Atom Eclipse and Sublime Text Senior Data Analyst Beroe Inc Chennai India May 2012 Dec 2016 Increased revenue by 40 by targeting the most profitable set of customers for a campaign about sustainability by performing a logistic regression technique Designed a product that provides actionable recommendations by identifying best cost sourcing suppliers LCCS for P G by making 95 accurate price forecasts in 2014-15 using elasticity modelling Projects Quantitative Analytics- Credit scoring model for loan applicants - Built a model to identify customers who were likely to default on a loan after extensive data cleaning-missing value and transforming the data outlier treatment . The model used was logistic regression with variables like total transactions purchase volume etc. Predictive Analytics Hospital Ranking - Analyzed hospital data and determined rank of all the hospitals in the United States of America based on the number of patients treated doctor availability and successful operations using python Skills Python SQL Hadoop Education University of Texas at Dallas M.S.in Data Analytics GPA 3.5 Jan 2017- Jul 2018 BITS Pilani B.E. in Engineering GPA 3.5 Aug 2007- May 2012 "

		# jd = 'Python'
		# resume = "java"

		print("\n jd --- ", jd)
		print("\n resume --- ", resume)
		similarity = ''
		try:
			jd_token = self.word_token(self.cleanText(jd))
			jd_token = [tpl[0] for tpl in pos_tag(jd_token) if tpl[1] in ['NN','VB'] ]
			# jd_token = [tpl[0] for tpl in pos_tag(jd_token) if tpl[1] in ['NN', 'NNS','NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] ]
			
			print("\n jd_token = ",jd_token)
			wmd_corpus = self.prepareData(resume)
			instance_wmd = WmdSimilarity(wmd_corpus, self.w2v_model)
			similarity = instance_wmd[jd_token][0]
			print("\n wmd sims --- ", similarity)
		except Exception as e:
			print("\n Error in getWmdSimilarity --- ", e, "\n", traceback.format_exc())
			pass
		return similarity

if __name__ == '__main__':
	obj = SemanticSimilarity()
	obj.getWmdSimilarity("netherland amsterdam", "html css")
