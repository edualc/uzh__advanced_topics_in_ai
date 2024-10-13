from speakeasypy import Speakeasy, Chatroom
from typing import List
import time
import rdflib
from decouple import config

import os
import sys
sys.path.append(os.getcwd())


listen_freq = 2


class BaseAgent:
    def __init__(self, graph, username, password):
        self.graph = graph
        self.username = username
        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=config('UZH_SPEAKEASY_HOST'), username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.

    def listen(self):
        raise NotImplementedError("Subclasses must implement this method.")
    

class AgentV1(BaseAgent):
    def listen(self):
        while True:
            # only check active chatrooms (i.e., remaining_time > 0) if active=True.
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    # send a welcome message if room is not initiated
                    room.post_messages(f'Hello! This is a welcome message from {room.my_alias}.')
                    room.initiated = True
                # Retrieve messages from this chat room.
                # If only_partner=True, it filters out messages sent by the current bot.
                # If only_new=True, it filters out messages that have already been marked as processed.
                for message in room.get_messages(only_partner=True, only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new message #{message.ordinal}: '{message.message}' "
                        f"- {self.get_time()}")

                    try:

                        # Implement your agent here #
                        query = message.message

                        
                        # Execute query on graph
                        qres = self.graph.query(query)

                        # Collect answers
                        # --> Currently, we do not add a LIMIT if there are too many rows returned (user issue)
                        all_answers = []
                        column_names = ', '.join(['?' + str(var) for var in qres.vars])

                        for row in qres:
                            answer = []
                            for col_name in qres.vars:
                                try:
                                    value = str(row[col_name])
                                except KeyError:
                                    value = None

                                answer.append(value)
                            all_answers.append(', '.join(answer))

                        # Send a message to the corresponding chat room using the post_messages method of the room object.
                        formatted_answer = '\n'.join([str(ans) for ans in all_answers])
                        room.post_messages(f"""
                                            Using the RDF Graph, the answer is:
                                            {column_names}
                                            {formatted_answer}
                                            """)
                        print(f"Answered message with:\n{formatted_answer}")


                    except Exception as e:
                        print(f"Error while processing message: {e}")
                        room.post_messages(f"Error while processing your message. Please try again.")
                        

                    # Mark the message as processed, so it will be filtered out when retrieving new messages.
                    room.mark_as_processed(message)

                # Retrieve reactions from this chat room.
                # If only_new=True, it filters out reactions that have already been marked as processed.
                for reaction in room.get_reactions(only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new reaction #{reaction.message_ordinal}: '{reaction.type}' "
                        f"- {self.get_time()}")

                    # Implement your agent here #

                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)

            time.sleep(listen_freq)

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    print(f"Loading graph")
    graph = rdflib.Graph()
    graph.parse('data/14_graph.nt', format='turtle') # change the path
    print(f"Graph loaded")

    demo_bot = AgentV1(graph, config("UZH_BOT_USERNAME"), config("UZH_BOT_PASSWORD"))
    demo_bot.listen()
