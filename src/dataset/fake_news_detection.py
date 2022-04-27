import requests
import re
import abc
import logging


class EvaluatedUser(object):

    def __init__(self, user_id, origin):
        """
        Constructor.
        :param user_id: The database id of the user
        :param origin: The platform that the user was fetched from
                        - "TWITTER"
                        - "REDDIT"
        """
        self.user_id = user_id
        self.origin = origin
        self.threads = []
        self.own_posts = []


class FakeNewsSpreaderDetector(abc.ABC):
    """
    Abstract detector for fake news spreading users.
    """
    @abc.abstractmethod
    def candidate(self, user, track_subreddit=False):
        pass

class AnnotatedFakeNewsDetector(FakeNewsSpreaderDetector):

    def __init__(self, domain_file_path, label):
        self.label = label
        self.urls = []
        self.bias_map = {}
        self.factuality_map = {}
        domain_file = open(domain_file_path, 'r')
        for annotation in domain_file.readlines():
            cells = annotation.split(';')
            domain = cells[0].strip()
            bias_lst = [bias.strip() for bias in cells[1].split(',')]
            factuality = cells[2].strip()
            self.bias_map[domain] = bias_lst
            self.factuality_map[domain] = factuality
            self.urls.append(domain)

    def candidate(self, user, content_index=2):
        post_map = {}
        for post in user.own_posts:
            links = LinkDetector.get_links_from_post(post[content_index])
            post_map[post[0]] = []
            visited_links = []
            for link in links:
                if link.endswith(')'):
                    link = link[:-1]
                if link in visited_links:
                    continue
                visited_links.append(link)

                for domain in self.urls:
                    if "." + domain in link.lower() or "/" + domain in link.lower():
                        post_map[post[0]].append((domain, self.label, self.bias_map[domain], self.factuality_map[domain]))
                        break
        return post_map


class LinkDetector(FakeNewsSpreaderDetector):
    """
    Implementation of a fake news detector
    that uses a list of domains to detect fake news spreaders
    based on the links in their posts.
    """

    def __init__(self, link_file_path='../data/domain_lists/fn_domains_verified'):
        """
        Constructor
        """
        self.link_file_path = link_file_path
        self.urls = []
        self.read_link_file()
        self.sr_map = {}
        self.domain_counter = {}

    def read_link_file(self):
        """
        Read the given file with the list
        of domains that are known to publish fake news
        :return:
        """
        with open(self.link_file_path) as f:
            lines = f.readlines()
            for line in lines:
                self.urls.append(line.rstrip())
        logging.info(self.urls)

    @staticmethod
    def demask_link(link):
        """
        Queue a shortened link to retrieve its real location
        Note: Time expensive
        :param link: The link to deshorten
        :return: The deshorted link
        """
        try:
            resp = requests.head(link)
            return resp.headers['Location']
        except Exception:
            return link

    @staticmethod
    def get_links_from_post(post):
        """
        Using a regular expression, extract all the links from a given post.
        :param post: The post to extract the links of
        :return: The list of found links
        """
        links = re.findall(
            r'(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))',
            post)
        return [link[0] for link in links]

    def candidate(self, user, track_subreddit=False, min_links=1, content_index=2):
        """
        Overrides the fake news predictor method
        """
        ids = set()
        for post in user.own_posts:
            links = LinkDetector.get_links_from_post(post[content_index])
            for link in links:
                """
                if "https://t.co" in link:
                    link = LinkDetector.demask_link(link)
                if "bit.ly" in link:
                    link = LinkDetector.demask_link(link)
                """
                for domain in self.urls:
                    domain = domain.replace('www.', '')
                    if "." + domain.lower() in link.lower() or "/" + domain.lower() in link.lower():
                        if domain in self.domain_counter.keys():
                            self.domain_counter[domain] = self.domain_counter[domain] + 1
                        else:
                            self.domain_counter[domain] = 1
                        ids.add(post[0])
                        if track_subreddit:
                            json_data = post[5]
                            post_subreddit = json_data['subreddit']
                            post_score = json_data['score']
                            logging.info(post_subreddit)
                            logging.info(post_score)
                            logging.info(link)
                            if post_subreddit in self.sr_map.keys():
                                self.sr_map[post_subreddit][0] += 1
                                self.sr_map[post_subreddit][1] += post_score
                            else:
                                self.sr_map[post_subreddit] = [1, post_score]
                        break

        return len(ids) >= min_links, ids