#include "greatest.h"

#include "event_loop.h"
#include "example_task.h"
#include "common.h"
#include "state/task_table.h"
#include "state/redis.h"

#include <unistd.h>

SUITE(task_table_tests);

static event_loop *g_loop;

/* ==== Test if operations time out correctly ==== */

/* === Test get task timeout === */

const char *lookup_timeout_context = "lookup_timeout";
int lookup_failed = 0;

void lookup_done_callback(task_spec *task,
                          void *context) {
  /* The done callback should not be called. */
  CHECK(0);
}

void lookup_fail_callback(unique_id id, void *user_data) {
  lookup_failed = 1;
  CHECK(user_data == (void *) lookup_timeout_context);
  event_loop_stop(g_loop);
}

TEST lookup_timeout_test(void) {
  g_loop = event_loop_create();
  db_handle *db =
      db_connect("127.0.0.1", 6379, "plasma_manager", "127.0.0.1", 1234);
  db_attach(db, g_loop);
  retry_info retry = {
      .num_retries = 5, .timeout = 100, .fail_callback = lookup_fail_callback,
  };
  task_table_get_task(db, NIL_ID, &retry, lookup_done_callback,
                      (void *) lookup_timeout_context);
  /* Disconnect the database to see if the lookup times out. */
  close(db->context->c.fd);
  event_loop_run(g_loop);
  db_disconnect(db);
  destroy_outstanding_callbacks(g_loop);
  event_loop_destroy(g_loop);
  ASSERT(lookup_failed);
  PASS();
}

/* === Test add timeout === */

const char *add_timeout_context = "add_timeout";
int add_failed = 0;

void add_done_callback(task_spec *task, void *context) {
  /* The done callback should not be called. */
  CHECK(0);
}

void add_fail_callback(unique_id id, void *user_data) {
  add_failed = 1;
  CHECK(user_data == (void *) add_timeout_context);
  event_loop_stop(g_loop);
}

TEST add_timeout_test(void) {
  g_loop = event_loop_create();
  db_handle *db =
      db_connect("127.0.0.1", 6379, "plasma_manager", "127.0.0.1", 1234);
  db_attach(db, g_loop);
  retry_info retry = {
      .num_retries = 5, .timeout = 100, .fail_callback = add_fail_callback,
  };
  task_table_get_task(db, NIL_ID, &retry, add_done_callback,
                   (void *) add_timeout_context);
  /* Disconnect the database to see if the lookup times out. */
  close(db->context->c.fd);
  event_loop_run(g_loop);
  db_disconnect(db);
  destroy_outstanding_callbacks(g_loop);
  event_loop_destroy(g_loop);
  ASSERT(add_failed);
  PASS();
}

/* ==== Test if the retry is working correctly ==== */

int64_t reconnect_context_callback(event_loop *loop,
                                   int64_t timer_id,
                                   void *context) {
  db_handle *db = context;
  /* Reconnect to redis. This is not reconnecting the pub/sub channel. */
  redisAsyncFree(db->context);
  redisFree(db->sync_context);
  db->context = redisAsyncConnect("127.0.0.1", 6379);
  db->context->data = (void *) db;
  db->sync_context = redisConnect("127.0.0.1", 6379);
  /* Re-attach the database to the event loop (the file descriptor changed). */
  db_attach(db, loop);
  return EVENT_LOOP_TIMER_DONE;
}

int64_t terminate_event_loop_callback(event_loop *loop,
                                      int64_t timer_id,
                                      void *context) {
  event_loop_stop(loop);
  return EVENT_LOOP_TIMER_DONE;
}

/* === Test lookup retry === */

const char *lookup_retry_context = "lookup_retry";
int lookup_retry_succeeded = 0;

void lookup_retry_done_callback(task_spec *task,
                                void *context) {
  CHECK(context == (void *) lookup_retry_context);
  lookup_retry_succeeded = 1;
}

void lookup_retry_fail_callback(unique_id id, void *user_data) {
  /* The fail callback should not be called. */
  CHECK(0);
}

TEST lookup_retry_test(void) {
  g_loop = event_loop_create();
  db_handle *db =
      db_connect("127.0.0.1", 6379, "plasma_manager", "127.0.0.1", 11235);
  db_attach(db, g_loop);
  retry_info retry = {
      .num_retries = 5,
      .timeout = 100,
      .fail_callback = lookup_retry_fail_callback,
  };
  task_table_get_task(db, NIL_ID, &retry, lookup_retry_done_callback,
                      (void *) lookup_retry_context);
  /* Disconnect the database to let the lookup time out the first time. */
  close(db->context->c.fd);
  /* Install handler for reconnecting the database. */
  event_loop_add_timer(
      g_loop, 150, (event_loop_timer_handler) reconnect_context_callback, db);
  /* Install handler for terminating the event loop. */
  event_loop_add_timer(g_loop, 750,
                       (event_loop_timer_handler) terminate_event_loop_callback,
                       NULL);
  event_loop_run(g_loop);
  db_disconnect(db);
  destroy_outstanding_callbacks(g_loop);
  event_loop_destroy(g_loop);
  ASSERT(lookup_retry_succeeded);
  PASS();
}

/* === Test add retry === */

const char *add_retry_context = "add_retry";
int add_retry_succeeded = 0;

void add_retry_done_callback(task_spec *task, void *user_context) {
  CHECK(user_context == (void *) add_retry_context);
  add_retry_succeeded = 1;
  free_task_spec(task);
}

void add_retry_fail_callback(unique_id id, void *user_data) {
  /* The fail callback should not be called. */
  CHECK(0);
}

TEST add_retry_test(void) {
  g_loop = event_loop_create();
  db_handle *db =
      db_connect("127.0.0.1", 6379, "plasma_manager", "127.0.0.1", 11235);
  db_attach(db, g_loop);
  retry_info retry = {
      .num_retries = 5,
      .timeout = 100,
      .fail_callback = add_retry_fail_callback,
  };
  task_spec *task = example_task();
  task_table_add_task(db, NIL_ID, task, &retry, add_retry_done_callback,
                   (void *) add_retry_context);
  /* Disconnect the database to let the add time out the first time. */
  close(db->context->c.fd);
  /* Install handler for reconnecting the database. */
  event_loop_add_timer(
      g_loop, 150, (event_loop_timer_handler) reconnect_context_callback, db);
  /* Install handler for terminating the event loop. */
  event_loop_add_timer(g_loop, 750,
                       (event_loop_timer_handler) terminate_event_loop_callback,
                       NULL);
  event_loop_run(g_loop);
  db_disconnect(db);
  destroy_outstanding_callbacks(g_loop);
  event_loop_destroy(g_loop);
  ASSERT(add_retry_succeeded);
  PASS();
}

/* ==== Test if late succeed is working correctly ==== */

/* === Test lookup late succeed === */

const char *lookup_late_context = "lookup_late";
int lookup_late_failed = 0;

void lookup_late_fail_callback(unique_id id, void *user_context) {
  CHECK(user_context == (void *) lookup_late_context);
  lookup_late_failed = 1;
}

void lookup_late_done_callback(task_spec *task,
                               void *context) {
  /* This function should never be called. */
  CHECK(0);
}

TEST lookup_late_test(void) {
  g_loop = event_loop_create();
  db_handle *db =
      db_connect("127.0.0.1", 6379, "plasma_manager", "127.0.0.1", 11236);
  db_attach(db, g_loop);
  retry_info retry = {
      .num_retries = 0,
      .timeout = 0,
      .fail_callback = lookup_late_fail_callback,
  };
  task_table_get_task(db, NIL_ID, &retry, lookup_late_done_callback,
                      (void *) lookup_late_context);
  /* Install handler for terminating the event loop. */
  event_loop_add_timer(g_loop, 750,
                       (event_loop_timer_handler) terminate_event_loop_callback,
                       NULL);
  /* First process timer events to make sure the timeout is processed before
   * anything else. */
  aeProcessEvents(g_loop, AE_TIME_EVENTS);
  event_loop_run(g_loop);
  db_disconnect(db);
  destroy_outstanding_callbacks(g_loop);
  event_loop_destroy(g_loop);
  ASSERT(lookup_late_failed);
  PASS();
}

/* === Test add late succeed === */

const char *add_late_context = "add_late";
int add_late_failed = 0;

void add_late_fail_callback(unique_id id, void *user_context) {
  CHECK(user_context == (void *) add_late_context);
  add_late_failed = 1;
}

void add_late_done_callback(task_spec *task, void *user_context) {
  /* This function should never be called. */
  CHECK(0);
}

TEST add_late_test(void) {
  g_loop = event_loop_create();
  db_handle *db =
      db_connect("127.0.0.1", 6379, "plasma_manager", "127.0.0.1", 11236);
  db_attach(db, g_loop);
  retry_info retry = {
      .num_retries = 0, .timeout = 0, .fail_callback = add_late_fail_callback,
  };
  task_spec *task = example_task();
  task_table_add_task(db, NIL_ID, task, &retry, add_late_done_callback,
                   (void *) add_late_context);
  /* Install handler for terminating the event loop. */
  event_loop_add_timer(g_loop, 750,
                       (event_loop_timer_handler) terminate_event_loop_callback,
                       NULL);
  /* First process timer events to make sure the timeout is processed before
   * anything else. */
  aeProcessEvents(g_loop, AE_TIME_EVENTS);
  event_loop_run(g_loop);
  db_disconnect(db);
  destroy_outstanding_callbacks(g_loop);
  event_loop_destroy(g_loop);
  ASSERT(add_late_failed);
  PASS();
}

SUITE(task_table_tests) {
  RUN_TEST(lookup_timeout_test);
  RUN_TEST(add_timeout_test);
  RUN_TEST(lookup_retry_test);
  RUN_TEST(add_retry_test);
  RUN_TEST(lookup_late_test);
  RUN_TEST(add_late_test);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(task_table_tests);
  GREATEST_MAIN_END();
}
