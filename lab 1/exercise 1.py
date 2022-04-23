import urllib.request, re

response = urllib.request.urlopen('https://www.centennialcollege.ca/programs-courses/full-time/artificial-intelligence-online/')
html = response.read().decode('utf-8')

title_regex = r'<title>(.*)</title>'
company_regex = r'<h3>Companies Offering Jobs</h3>\n<p>(.*)</p>'
career_regex = r'<h3>Career Outlook</h3>\n[\s]*<ul[^>]*>\s*(?:<li[^>]*>(?:[^<]+<[^li])+li>\s*)+<\/ul>'
title = re.findall(title_regex, html)[0]
company = re.findall(company_regex, html)[0]
career = re.findall(career_regex, html)
career_regex = r'<li>(.*)</li>'
career = re.findall(career_regex, career[0])

file = open('terry my future', 'w')
file.write('title' + '\n' + title)
file.write('\ncompanies' + '\n' + company)
file.write('\ncareers\n')
for item in career:
	file.write(item + '\n')
file.close()