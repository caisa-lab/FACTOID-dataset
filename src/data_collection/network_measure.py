import itertools
import networkx as nx
import community as community_louvain
from networkx.algorithms.community.label_propagation import asyn_lpa_communities
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
import logging

# Users Table indexes
DB_ID_INDEX = 0

# Posts Table indexes
POST_ID_INDEX = 0
POST_AUTHOR_INDEX = 1
POST_CONTENT_INDEX = 2
POST_PARENT_INDEX = 4
POST_STANCE_INDEX = 7
POST_TIME_STAMP_INDEX = 8
POST_TOPIC_INDEX = 9
POST_TOPLEVEL_INDEX = 10
POST_ORIGINSPECIFIC_INDEX = 5


class NetworkMeasure(object):
    """
    Class for extracting the network structure of the users
    interaction (around o certain topic) from the dasp database.
    """

    def __init__(self, origin):
        """
        Constructor
        :param origin: Origin of the network to extract
                       - "TWITTER"
                       - "REDDIT"
        """
        self.origin = origin
        self.db = None
        self.network = None
        self.partition = None
        self.cmap = None
        self.node_users = []
        self.user_post_map = {}
        self.hex_colors = []


    def get_subreddit_users(self, subreddits):
        """
        Only for REDDIT:
        Filter out the user ids of the users that interacted in one of
        a given list of subreddits
        :param subreddits: The subreddits to filter
        :return: The list of found user ids
        """
        if not self.origin == "REDDIT":
            raise Exception("Filtering for subreddits is only available when working with data from reddit")
        else:
            try:
                self.db = OpinionDatabase()
                posts_full = self.all_posts_queue()
                filtered_users = set()
                for post in posts_full:
                    json_data = post[POST_ORIGINSPECIFIC_INDEX]
                    subreddit = json_data['subreddit']
                    if subreddit in subreddits:
                        filtered_users.add(post[POST_AUTHOR_INDEX])
                self.db.close_connection()
                return filtered_users
            except Exception as e:
                logging.warning(e)
                self.db.close_connection()
                raise Exception("Error during database queues!")

    def get_topic_users(self, topic):
        """
        Get a list of users that interacted in discussion around a certain topic.
        :param topic: The topic to filter for
        :return: The found list of user ids
        """
        try:
            self.db = OpinionDatabase()
            posts_full = self.all_posts_queue()
            topic_users = self.filter_topic_users(topic, posts_full)
            self.db.close_connection()
            return topic_users
        except Exception as e:
            logging.warning(e)
            self.db.close_connection()
            raise Exception("Error during database queues!")

    def build_network(self, filter_users=None, topic_filter=None, post_amount_filter=5, time_frame_filter=(None, None)):
        """
        Build the networkX user network based on the loaded thread map.
        The nodes (users) can filtered with some constraints.
        :param filter_users: Filter by a certain lsit of user ids
        :param topic_filter: Filter by users that interacted in a certain topic
        """
        print("Building social network graph...")
        try:
            self.db = OpinionDatabase()
            toplevel_posts = self.toplevel_posts_queue(time_frame=time_frame_filter)
            posts_full = self.all_posts_queue(time_frame=time_frame_filter)

            thread_mapping, unmatched_posts = NetworkMeasure.build_thread_mapping(toplevel_posts, posts_full)
            post_amount_map = self.post_amount_queue()
            self.user_post_map = NetworkMeasure.build_user_post_map(posts_full)
            author_set = set()  # Will represent the nodes of the network
            self.db.close_connection()
        except Exception as e:
            print(e)
            self.db.close_connection()
            raise Exception("Error during database queues!")

        edge_weights = {}
        for thread in thread_mapping.values():
            thread_participants = []
            topic_thread = False
            for post in thread:
                if post[POST_AUTHOR_INDEX] not in thread_participants and not post[POST_AUTHOR_INDEX] == '[deleted]':
                    thread_participants.append(post[POST_AUTHOR_INDEX])
                    if topic_filter is not None and post[POST_TOPIC_INDEX] == topic_filter:
                        topic_thread = True
            if topic_filter is not None and not topic_thread:
                continue
            if filter_users is not None:
                thread_participants = NetworkMeasure.filter_users(filter_users, thread_participants)
            thread_participants = NetworkMeasure.filter_users_amount(thread_participants, post_amount_map,
                                                                     threshold=post_amount_filter)
            author_set.update(thread_participants)
            raw_edges = get_network_edges(thread)
            for edge in raw_edges:
                if edge[0] in thread_participants and edge[1] in thread_participants:
                    if edge in edge_weights.keys():
                        edge_weights[edge] += 1
                    elif (edge[1], edge[0]) in edge_weights.keys():
                        edge_weights[(edge[1], edge[0])] += 1
                    else:
                        edge_weights[edge] = 1

        # This block needed to be added for the case where due to timeframed post fetching,
        # there are threads without a toplevel post in the timeframe fetched from the db
        if time_frame_filter[0] is not None and time_frame_filter[1] is not None:
            visited = []
            # Sort by timestamp (latest first) to rebuild the threads from the bottom up
            unmatched_posts = {k: v for k, v in reversed(
                sorted(unmatched_posts.items(), key=lambda item: item[1][POST_TIME_STAMP_INDEX]))}
            for pid, post in unmatched_posts.items():
                if pid not in visited:
                    thread_authors = set()
                    thread_authors.update(post[POST_AUTHOR_INDEX])
                    curr_post = post
                    visited.append(curr_post[POST_ID_INDEX])
                    parent_id = post[POST_PARENT_INDEX]
                    while parent_id in unmatched_posts.keys():
                        curr_post = unmatched_posts[curr_post[POST_PARENT_INDEX]]
                        thread_authors.update(curr_post[POST_AUTHOR_INDEX])
                        visited.append(curr_post[POST_ID_INDEX])
                        parent_id = curr_post[POST_PARENT_INDEX]
                    # Add authors
                    author_set.update(thread_authors)
                    # Add edges
                    raw_edges = itertools.combinations(thread_authors, 2)
                    for edge in raw_edges:
                        if edge[0] in thread_authors and edge[1] in thread_authors:
                            if edge in edge_weights.keys():
                                edge_weights[edge] += 1
                            elif (edge[1], edge[0]) in edge_weights.keys():
                                edge_weights[(edge[1], edge[0])] += 1
                            else:
                                edge_weights[edge] = 1

        logging.info(len(edge_weights))
        graph = nx.Graph()
        graph.add_nodes_from(author_set)
        for edge, weight_val in edge_weights.items():
            graph.add_edge(edge[0], edge[1], weight=weight_val)
        self.network = graph
        self.node_users = author_set
        logging.info(nx.info(self.network))

    def extract_largest_connected_component(self):
        """
        Extract the largest connected component in the networkX network.
        """
        for comp in nx.connected_components(self.network):
            if len(comp) > 5:
                logging.info("Component: " + str(len(comp)))

        user_set = max(nx.connected_components(self.network), key=len)
        self.network = self.network.subgraph(user_set)

    def color_network_nodes(self, profile='community_fluid'):
        """
        Color the nodes of the network with a certain
        strategy/profile:
        Most of them are community detection algorithms.
        Available profiles:
            - "community_louvain"
            - "community_max_modularity"
            - "community_propagation"
            - "community_fluid"
            - "gender"
            - "age_group"
            - "race"
        :param profile: The strategy to color the network nodes by
        """
        if profile == 'community_louvain':
            self.partition = community_louvain.best_partition(self.network)
        elif profile == 'community_max_modularity':
            sets = greedy_modularity_communities(self.network, weight='weight')
            mapping = {}

            for index, com_set in enumerate(sets):
                for uid in com_set:
                    mapping[uid] = index

            test_count = {}
            for user, cluster in mapping.items():
                if cluster in test_count.keys():
                    test_count[cluster] += 1
                else:
                    test_count[cluster] = 1
            logging.info([num for num in test_count.values() if num > 1])
            self.partition = mapping
        elif profile == 'community_propagation':
            sets = asyn_lpa_communities(self.network, "weight")
            mapping = {}

            for index, com_set in enumerate(sets):
                for uid in com_set:
                    mapping[uid] = index

            test_count = {}
            for user, cluster in mapping.items():
                if cluster in test_count.keys():
                    test_count[cluster] += 1
                else:
                    test_count[cluster] = 1
            logging.info([num for num in test_count.values() if num > 1])
            self.partition = mapping
        elif profile == 'community_fluid':
            best_partition = None
            best_score = 0

            for iteration in range(5):
                sets = asyn_fluidc(self.network, 3)
                mapping = {}

                for index, com_set in enumerate(sets):
                    for uid in com_set:
                        mapping[uid] = index

                test_count = {}
                for user, cluster in mapping.items():
                    if cluster in test_count.keys():
                        test_count[cluster] += 1
                    else:
                        test_count[cluster] = 1
                logging.info([num for num in test_count.values() if num > 1])

                score, per_cluster = self.score_partition(mapping)
                if score > best_score:
                    best_partition = mapping
                    best_score = score

            self.partition = best_partition
        elif profile == 'unicolor':
            mapping = {}
            for uid in list(self.network.nodes.keys()):
                mapping[uid] = 1
            self.partition = mapping
        else:
            raise Exception("Invalid profiling choice")

        self.cmap = cm.get_cmap('viridis', (max(self.partition.values()) + 1))
        colors = []
        for rgb in self.cmap.colors:
            colors.append(rgb_to_hex((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))))
        self.hex_colors = colors

    def set_partition(self, partition):
        mapping = {}
        for uid in list(self.network.nodes.keys()):
            mapping[uid] = partition[uid]
        self.partition = mapping
        self.cmap = cm.get_cmap('viridis', (max(self.partition.values()) + 1))
        colors = []
        for rgb in self.cmap.colors:
            colors.append(rgb_to_hex((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))))
        self.hex_colors = colors
        print(self.hex_colors)

    def get_partition_as_list(self):
        """
        Returns the current network partition
        as a list of list of user ids.
        :return: The current network partition
        as a list of list of user ids.
        """
        collector = {}

        for user, assignment in self.partition.items():
            if assignment in collector.keys():
                collector[assignment].add(user)
            else:
                new_set = set()
                new_set.add(user)
                collector[assignment] = new_set

        for cluster in collector.keys():
            if not self.hex_colors:
                break
            logging.info(self.hex_colors[cluster])

        return list(collector.values())

    def get_color_of_class(self, class_id):
        rgb = self.cmap(class_id)
        return rgb_to_hex((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

    def visualize_network(self):
        """
        Visualize the loaded network with pyplot.
        Can also be saved based on this.
        """
        layout = nx.spring_layout(self.network)
        plt.axis("off")
        nx.draw_networkx_nodes(self.network, layout, self.partition.keys(), node_size=15,
                               cmap=self.cmap, node_color=list(self.partition.values()))
        nx.draw_networkx_edges(self.network, layout, alpha=0.5)

    def score_partition(self, cluster_map, do_print=False):
        """
        Score the given partition with the
        community scores separability and density.
        :param cluster_map: The partition ot score
        :param do_print: Whether to print out hte results in the console
        :return: The average separability, tuples of the intermediate scores (separability,density)
        """
        """
        clean_edges = []
        for edge in self.network.edges:
            print("edge")
            if (edge[0], edge[1]) not in clean_edges and (edge[1], edge[0]) not in clean_edges:
                clean_edges.append((edge[0], edge[1]))"""
        clean_edges = self.network.edges()
        inter_edges = {}
        intra_edges = {}
        node_count = {}

        for node, community in cluster_map.items():
            if node not in self.network:
                continue
            if community not in inter_edges.keys():
                inter_edges[community] = 0
                intra_edges[community] = 0

            if community in node_count.keys():
                node_count[community] += 1
            else:
                node_count[community] = 1

        for edge in clean_edges:
            if cluster_map[edge[0]] == cluster_map[edge[1]]:
                intra_edges[cluster_map[edge[0]]] += 1
            else:
                inter_edges[cluster_map[edge[0]]] += 1
                inter_edges[cluster_map[edge[1]]] += 1

        sep_sum = 0
        scores = []
        for community, inter in inter_edges.items():
            nodes = node_count[community]
            if do_print:
                print(self.hex_colors[community])
                print("Separability:")
                print(intra_edges[community] / inter)
                print("Density:")
                print(intra_edges[community] / (nodes * (nodes - 1) / 2))

            scores.append((intra_edges[community] / inter, intra_edges[community] / (nodes * (nodes - 1) / 2)))
            sep_sum += intra_edges[community] / inter

        return sep_sum / len(node_count.keys()), scores
